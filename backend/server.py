from fastapi import FastAPI, APIRouter, UploadFile, File, HTTPException, Query
from fastapi.responses import StreamingResponse
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional, Dict, Any
import uuid
from datetime import datetime, timezone
import pandas as pd
import io

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

# Create the main app
app = FastAPI(title="PharmaAudit Pro API")

# Create a router with the /api prefix
api_router = APIRouter(prefix="/api")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ==================== MODELS ====================

# Store Models
class Store(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    code: str
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

class StoreResponse(BaseModel):
    id: str
    name: str
    code: str
    created_at: str

# Stock Master Models
class StockItem(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    store_id: str
    store_name: str
    product_name: str
    manufacturer: str
    batch_number: str
    expiry_date: str
    mrp: float
    system_quantity: int
    rack_code: str
    upload_id: str
    uploaded_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

# Audit Models
class AuditCreate(BaseModel):
    store_id: str
    store_name: str
    audit_type: str  # 'rack' or 'manufacturer'
    reference_value: str  # rack_code or manufacturer name
    employee_name: str
    items: List[Dict[str, Any]]

class AuditItemInput(BaseModel):
    stock_item_id: str
    physical_quantity: int
    physical_batch: str
    physical_expiry: str
    physical_mrp: float
    status: str = "audited"  # audited, not_found, damaged, expired
    notes: Optional[str] = None

class AuditSubmit(BaseModel):
    items: List[AuditItemInput]

class Audit(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    store_id: str
    store_name: str
    audit_type: str
    reference_value: str
    employee_name: str
    status: str = "in_progress"  # in_progress, completed, released
    total_items: int
    audited_items: int = 0
    started_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    completed_at: Optional[str] = None

class AuditItem(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    audit_id: str
    stock_item_id: str
    product_name: str
    manufacturer: str
    rack_code: str
    # System values
    system_batch: str
    system_expiry: str
    system_mrp: float
    system_quantity: int
    # Physical values
    physical_quantity: Optional[int] = None
    physical_batch: Optional[str] = None
    physical_expiry: Optional[str] = None
    physical_mrp: Optional[float] = None
    # Deviations
    quantity_difference: Optional[int] = None
    has_quantity_deviation: bool = False
    has_batch_deviation: bool = False
    has_expiry_deviation: bool = False
    has_mrp_deviation: bool = False
    # Status
    status: str = "pending"  # pending, audited, not_found, damaged, expired
    notes: Optional[str] = None
    audited_at: Optional[str] = None

# Audit Marker Models
class AuditMarker(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    store_id: str
    store_name: str
    audit_type: str
    reference_id: str  # rack_code or manufacturer name
    audited_on: str
    audited_by: str
    status: str = "locked"  # locked, released
    audit_id: str

# Upload History
class UploadHistory(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    filename: str
    uploaded_by: str
    uploaded_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    total_records: int
    status: str = "success"

# ==================== PREDEFINED STORES ====================

PREDEFINED_STORES = [
    {"name": "HANAMAKONDA", "code": "HNK"},
    {"name": "MG ROAD", "code": "MGR"},
    {"name": "RAMNAGAR", "code": "RMN"},
    {"name": "CHOWRASTHA", "code": "CHW"},
    {"name": "SUBEDARI", "code": "SBD"},
    {"name": "WAREHOUSE", "code": "WRH"},
    {"name": "FORT ROAD", "code": "FRT"},
]

# ==================== ROUTES ====================

@api_router.get("/")
async def root():
    return {"message": "PharmaAudit Pro API", "version": "1.0.0"}

# Store Routes
@api_router.post("/stores/init", response_model=List[StoreResponse])
async def initialize_stores():
    """Initialize predefined stores if they don't exist"""
    existing = await db.stores.count_documents({})
    if existing > 0:
        stores = await db.stores.find({}, {"_id": 0}).to_list(100)
        return stores
    
    stores_to_insert = []
    for store_data in PREDEFINED_STORES:
        store = Store(name=store_data["name"], code=store_data["code"])
        stores_to_insert.append(store.model_dump())
    
    await db.stores.insert_many(stores_to_insert)
    return stores_to_insert

@api_router.get("/stores", response_model=List[StoreResponse])
async def get_stores():
    """Get all stores"""
    stores = await db.stores.find({}, {"_id": 0}).to_list(100)
    if not stores:
        # Auto-initialize if empty
        return await initialize_stores()
    return stores

# Stock Upload Routes
@api_router.get("/stock/template")
async def download_template():
    """Download Excel template for stock upload"""
    df = pd.DataFrame({
        "Store Name": ["HANAMAKONDA", "MG ROAD"],
        "Product Name": ["Paracetamol 500mg", "Amoxicillin 250mg"],
        "Manufacturer": ["Sun Pharma", "Cipla"],
        "Batch Number": ["BN001", "BN002"],
        "Expiry Date": ["2025-12-31", "2026-06-30"],
        "MRP": [25.50, 120.00],
        "System Quantity": [100, 50],
        "Rack Code": ["R1", "R2"]
    })
    
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Stock')
    output.seek(0)
    
    return StreamingResponse(
        output,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": "attachment; filename=stock_template.xlsx"}
    )

@api_router.post("/stock/upload")
async def upload_stock(
    file: UploadFile = File(...),
    employee_name: str = Query(...)
):
    """Upload stock Excel file"""
    if not file.filename.endswith(('.xlsx', '.xls')):
        raise HTTPException(status_code=400, detail="Only Excel files are allowed")
    
    try:
        contents = await file.read()
        df = pd.read_excel(io.BytesIO(contents))
        
        required_columns = [
            "Store Name", "Product Name", "Manufacturer", 
            "Batch Number", "Expiry Date", "MRP", 
            "System Quantity", "Rack Code"
        ]
        
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            raise HTTPException(
                status_code=400, 
                detail=f"Missing required columns: {', '.join(missing)}"
            )
        
        # Get store mapping
        stores = await db.stores.find({}, {"_id": 0}).to_list(100)
        store_map = {s["name"].upper(): s["id"] for s in stores}
        
        upload_id = str(uuid.uuid4())
        stock_items = []
        
        for _, row in df.iterrows():
            store_name = str(row["Store Name"]).strip().upper()
            store_id = store_map.get(store_name)
            
            if not store_id:
                continue  # Skip unknown stores
            
            # Handle expiry date
            expiry = row["Expiry Date"]
            if isinstance(expiry, pd.Timestamp):
                expiry_str = expiry.strftime("%Y-%m-%d")
            else:
                expiry_str = str(expiry)
            
            item = StockItem(
                store_id=store_id,
                store_name=store_name,
                product_name=str(row["Product Name"]).strip(),
                manufacturer=str(row["Manufacturer"]).strip(),
                batch_number=str(row["Batch Number"]).strip(),
                expiry_date=expiry_str,
                mrp=float(row["MRP"]),
                system_quantity=int(row["System Quantity"]),
                rack_code=str(row["Rack Code"]).strip().upper(),
                upload_id=upload_id
            )
            stock_items.append(item.model_dump())
        
        if stock_items:
            await db.stock_master.insert_many(stock_items)
        
        # Record upload history
        upload_record = UploadHistory(
            id=upload_id,
            filename=file.filename,
            uploaded_by=employee_name,
            total_records=len(stock_items)
        )
        await db.upload_history.insert_one(upload_record.model_dump())
        
        return {
            "message": "Stock uploaded successfully",
            "upload_id": upload_id,
            "total_records": len(stock_items)
        }
        
    except Exception as e:
        logger.error(f"Error uploading stock: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/stock/uploads")
async def get_upload_history():
    """Get upload history"""
    uploads = await db.upload_history.find({}, {"_id": 0}).sort("uploaded_at", -1).to_list(100)
    return uploads

@api_router.get("/stock/{store_id}/racks")
async def get_store_racks(store_id: str):
    """Get unique rack codes for a store"""
    pipeline = [
        {"$match": {"store_id": store_id}},
        {"$group": {"_id": "$rack_code", "count": {"$sum": 1}}},
        {"$sort": {"_id": 1}}
    ]
    results = await db.stock_master.aggregate(pipeline).to_list(1000)
    
    # Get audit markers for this store
    markers = await db.audit_markers.find(
        {"store_id": store_id, "audit_type": "rack"},
        {"_id": 0}
    ).to_list(1000)
    marker_map = {m["reference_id"]: m for m in markers}
    
    racks = []
    for r in results:
        rack_code = r["_id"]
        marker = marker_map.get(rack_code)
        racks.append({
            "rack_code": rack_code,
            "item_count": r["count"],
            "is_audited": marker is not None and marker.get("status") == "locked",
            "is_released": marker is not None and marker.get("status") == "released",
            "audited_by": marker.get("audited_by") if marker else None,
            "audited_on": marker.get("audited_on") if marker else None
        })
    
    return racks

@api_router.get("/stock/{store_id}/manufacturers")
async def get_store_manufacturers(store_id: str):
    """Get unique manufacturers for a store"""
    pipeline = [
        {"$match": {"store_id": store_id}},
        {"$group": {"_id": "$manufacturer", "count": {"$sum": 1}}},
        {"$sort": {"_id": 1}}
    ]
    results = await db.stock_master.aggregate(pipeline).to_list(1000)
    
    # Get audit markers for this store
    markers = await db.audit_markers.find(
        {"store_id": store_id, "audit_type": "manufacturer"},
        {"_id": 0}
    ).to_list(1000)
    marker_map = {m["reference_id"]: m for m in markers}
    
    manufacturers = []
    for r in results:
        mfr = r["_id"]
        marker = marker_map.get(mfr)
        manufacturers.append({
            "manufacturer": mfr,
            "item_count": r["count"],
            "is_audited": marker is not None and marker.get("status") == "locked",
            "is_released": marker is not None and marker.get("status") == "released",
            "audited_by": marker.get("audited_by") if marker else None,
            "audited_on": marker.get("audited_on") if marker else None
        })
    
    return manufacturers

@api_router.get("/stock/{store_id}/items")
async def get_store_items(
    store_id: str,
    rack_code: Optional[str] = None,
    manufacturer: Optional[str] = None
):
    """Get stock items filtered by rack or manufacturer"""
    query = {"store_id": store_id}
    if rack_code:
        query["rack_code"] = rack_code.upper()
    if manufacturer:
        query["manufacturer"] = manufacturer
    
    items = await db.stock_master.find(query, {"_id": 0}).to_list(10000)
    return items

# Audit Routes
@api_router.post("/audits")
async def create_audit(audit_data: AuditCreate):
    """Create a new audit session"""
    # Check if already audited and not released
    existing_marker = await db.audit_markers.find_one({
        "store_id": audit_data.store_id,
        "audit_type": audit_data.audit_type,
        "reference_id": audit_data.reference_value,
        "status": "locked"
    }, {"_id": 0})
    
    if existing_marker:
        raise HTTPException(
            status_code=400,
            detail="This audit task is already completed. Contact admin to release for re-audit."
        )
    
    # Get stock items for this audit
    query = {"store_id": audit_data.store_id}
    if audit_data.audit_type == "rack":
        query["rack_code"] = audit_data.reference_value.upper()
    else:
        query["manufacturer"] = audit_data.reference_value
    
    stock_items = await db.stock_master.find(query, {"_id": 0}).to_list(10000)
    
    if not stock_items:
        raise HTTPException(status_code=404, detail="No items found for this audit task")
    
    # Create audit
    audit = Audit(
        store_id=audit_data.store_id,
        store_name=audit_data.store_name,
        audit_type=audit_data.audit_type,
        reference_value=audit_data.reference_value,
        employee_name=audit_data.employee_name,
        total_items=len(stock_items)
    )
    
    await db.audits.insert_one(audit.model_dump())
    
    # Create audit items
    audit_items = []
    for item in stock_items:
        audit_item = AuditItem(
            audit_id=audit.id,
            stock_item_id=item["id"],
            product_name=item["product_name"],
            manufacturer=item["manufacturer"],
            rack_code=item["rack_code"],
            system_batch=item["batch_number"],
            system_expiry=item["expiry_date"],
            system_mrp=item["mrp"],
            system_quantity=item["system_quantity"]
        )
        audit_items.append(audit_item.model_dump())
    
    await db.audit_items.insert_many(audit_items)
    
    return {
        "audit_id": audit.id,
        "total_items": len(stock_items),
        "message": "Audit started successfully"
    }

@api_router.get("/audits")
async def get_audits(
    store_id: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = 100
):
    """Get all audits"""
    query = {}
    if store_id:
        query["store_id"] = store_id
    if status:
        query["status"] = status
    
    audits = await db.audits.find(query, {"_id": 0}).sort("started_at", -1).to_list(limit)
    return audits

@api_router.get("/audits/{audit_id}")
async def get_audit(audit_id: str):
    """Get audit details with items"""
    audit = await db.audits.find_one({"id": audit_id}, {"_id": 0})
    if not audit:
        raise HTTPException(status_code=404, detail="Audit not found")
    
    items = await db.audit_items.find({"audit_id": audit_id}, {"_id": 0}).to_list(10000)
    
    return {
        "audit": audit,
        "items": items
    }

@api_router.put("/audits/{audit_id}/items/{item_id}")
async def update_audit_item(audit_id: str, item_id: str, item_data: AuditItemInput):
    """Update a single audit item"""
    audit_item = await db.audit_items.find_one({"id": item_id, "audit_id": audit_id}, {"_id": 0})
    if not audit_item:
        raise HTTPException(status_code=404, detail="Audit item not found")
    
    # Calculate deviations
    quantity_diff = item_data.physical_quantity - audit_item["system_quantity"]
    has_qty_dev = quantity_diff != 0
    has_batch_dev = item_data.physical_batch.strip().upper() != audit_item["system_batch"].strip().upper()
    has_expiry_dev = item_data.physical_expiry != audit_item["system_expiry"]
    has_mrp_dev = abs(item_data.physical_mrp - audit_item["system_mrp"]) > 0.01
    
    update_data = {
        "physical_quantity": item_data.physical_quantity,
        "physical_batch": item_data.physical_batch,
        "physical_expiry": item_data.physical_expiry,
        "physical_mrp": item_data.physical_mrp,
        "quantity_difference": quantity_diff,
        "has_quantity_deviation": has_qty_dev,
        "has_batch_deviation": has_batch_dev,
        "has_expiry_deviation": has_expiry_dev,
        "has_mrp_deviation": has_mrp_dev,
        "status": item_data.status,
        "notes": item_data.notes,
        "audited_at": datetime.now(timezone.utc).isoformat()
    }
    
    await db.audit_items.update_one(
        {"id": item_id},
        {"$set": update_data}
    )
    
    # Update audit progress
    audited_count = await db.audit_items.count_documents({
        "audit_id": audit_id,
        "status": {"$ne": "pending"}
    })
    
    await db.audits.update_one(
        {"id": audit_id},
        {"$set": {"audited_items": audited_count}}
    )
    
    return {"message": "Item updated", "audited_count": audited_count}

@api_router.post("/audits/{audit_id}/submit")
async def submit_audit(audit_id: str):
    """Submit completed audit"""
    audit = await db.audits.find_one({"id": audit_id}, {"_id": 0})
    if not audit:
        raise HTTPException(status_code=404, detail="Audit not found")
    
    # Check if all items are audited
    pending_count = await db.audit_items.count_documents({
        "audit_id": audit_id,
        "status": "pending"
    })
    
    if pending_count > 0:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot submit. {pending_count} items are still pending audit."
        )
    
    # Update audit status
    completed_at = datetime.now(timezone.utc).isoformat()
    await db.audits.update_one(
        {"id": audit_id},
        {"$set": {"status": "completed", "completed_at": completed_at}}
    )
    
    # Create/Update audit marker
    marker = AuditMarker(
        store_id=audit["store_id"],
        store_name=audit["store_name"],
        audit_type=audit["audit_type"],
        reference_id=audit["reference_value"],
        audited_on=completed_at,
        audited_by=audit["employee_name"],
        status="locked",
        audit_id=audit_id
    )
    
    # Remove any existing released marker
    await db.audit_markers.delete_many({
        "store_id": audit["store_id"],
        "audit_type": audit["audit_type"],
        "reference_id": audit["reference_value"]
    })
    
    await db.audit_markers.insert_one(marker.model_dump())
    
    return {"message": "Audit submitted successfully", "audit_id": audit_id}

# Audit Markers Routes
@api_router.get("/audit-markers")
async def get_audit_markers(
    store_id: Optional[str] = None,
    audit_type: Optional[str] = None
):
    """Get audit markers"""
    query = {}
    if store_id:
        query["store_id"] = store_id
    if audit_type:
        query["audit_type"] = audit_type
    
    markers = await db.audit_markers.find(query, {"_id": 0}).to_list(1000)
    return markers

@api_router.put("/audit-markers/{marker_id}/release")
async def release_audit_marker(marker_id: str):
    """Release an audit for re-audit"""
    result = await db.audit_markers.update_one(
        {"id": marker_id},
        {"$set": {"status": "released"}}
    )
    
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Marker not found")
    
    return {"message": "Audit released for re-audit"}

# Reports Routes
@api_router.get("/reports/summary")
async def get_reports_summary(store_id: Optional[str] = None):
    """Get audit summary report"""
    match_stage = {}
    if store_id:
        match_stage["store_id"] = store_id
    
    # Total audits
    total_audits = await db.audits.count_documents(match_stage)
    completed_audits = await db.audits.count_documents({**match_stage, "status": "completed"})
    
    # Deviation summary
    pipeline = [
        {"$match": {"status": {"$ne": "pending"}}},
        {"$group": {
            "_id": None,
            "total_items": {"$sum": 1},
            "qty_deviations": {"$sum": {"$cond": ["$has_quantity_deviation", 1, 0]}},
            "batch_deviations": {"$sum": {"$cond": ["$has_batch_deviation", 1, 0]}},
            "expiry_deviations": {"$sum": {"$cond": ["$has_expiry_deviation", 1, 0]}},
            "mrp_deviations": {"$sum": {"$cond": ["$has_mrp_deviation", 1, 0]}},
            "shortage_value": {"$sum": {
                "$cond": [
                    {"$lt": ["$quantity_difference", 0]},
                    {"$multiply": [{"$abs": "$quantity_difference"}, "$system_mrp"]},
                    0
                ]
            }},
            "excess_value": {"$sum": {
                "$cond": [
                    {"$gt": ["$quantity_difference", 0]},
                    {"$multiply": ["$quantity_difference", "$system_mrp"]},
                    0
                ]
            }}
        }}
    ]
    
    deviation_stats = await db.audit_items.aggregate(pipeline).to_list(1)
    stats = deviation_stats[0] if deviation_stats else {
        "total_items": 0,
        "qty_deviations": 0,
        "batch_deviations": 0,
        "expiry_deviations": 0,
        "mrp_deviations": 0,
        "shortage_value": 0,
        "excess_value": 0
    }
    
    return {
        "total_audits": total_audits,
        "completed_audits": completed_audits,
        "in_progress_audits": total_audits - completed_audits,
        **stats
    }

@api_router.get("/reports/store-wise")
async def get_store_wise_report():
    """Get store-wise audit completion report"""
    stores = await db.stores.find({}, {"_id": 0}).to_list(100)
    
    report = []
    for store in stores:
        total_racks = await db.stock_master.distinct("rack_code", {"store_id": store["id"]})
        total_mfrs = await db.stock_master.distinct("manufacturer", {"store_id": store["id"]})
        
        audited_racks = await db.audit_markers.count_documents({
            "store_id": store["id"],
            "audit_type": "rack",
            "status": "locked"
        })
        
        audited_mfrs = await db.audit_markers.count_documents({
            "store_id": store["id"],
            "audit_type": "manufacturer",
            "status": "locked"
        })
        
        report.append({
            "store_id": store["id"],
            "store_name": store["name"],
            "total_racks": len(total_racks),
            "audited_racks": audited_racks,
            "pending_racks": len(total_racks) - audited_racks,
            "total_manufacturers": len(total_mfrs),
            "audited_manufacturers": audited_mfrs,
            "pending_manufacturers": len(total_mfrs) - audited_mfrs
        })
    
    return report

@api_router.get("/reports/deviations")
async def get_deviations_report(
    deviation_type: str = Query(..., description="qty, batch, expiry, mrp"),
    store_id: Optional[str] = None
):
    """Get detailed deviation report"""
    match_query = {"status": {"$ne": "pending"}}
    
    if deviation_type == "qty":
        match_query["has_quantity_deviation"] = True
    elif deviation_type == "batch":
        match_query["has_batch_deviation"] = True
    elif deviation_type == "expiry":
        match_query["has_expiry_deviation"] = True
    elif deviation_type == "mrp":
        match_query["has_mrp_deviation"] = True
    
    items = await db.audit_items.find(match_query, {"_id": 0}).to_list(10000)
    
    # Enrich with audit info
    audit_ids = list(set(item["audit_id"] for item in items))
    audits = await db.audits.find({"id": {"$in": audit_ids}}, {"_id": 0}).to_list(1000)
    audit_map = {a["id"]: a for a in audits}
    
    if store_id:
        items = [i for i in items if audit_map.get(i["audit_id"], {}).get("store_id") == store_id]
    
    for item in items:
        audit = audit_map.get(item["audit_id"], {})
        item["store_name"] = audit.get("store_name", "")
        item["audited_by"] = audit.get("employee_name", "")
    
    return items

@api_router.get("/reports/export")
async def export_report(
    report_type: str = Query(..., description="summary, deviations, store-wise"),
    format: str = Query("excel", description="excel or csv"),
    store_id: Optional[str] = None,
    deviation_type: Optional[str] = None
):
    """Export reports as Excel or CSV"""
    if report_type == "deviations" and deviation_type:
        data = await get_deviations_report(deviation_type, store_id)
    elif report_type == "store-wise":
        data = await get_store_wise_report()
    else:
        data = [await get_reports_summary(store_id)]
    
    df = pd.DataFrame(data)
    output = io.BytesIO()
    
    if format == "csv":
        df.to_csv(output, index=False)
        output.seek(0)
        return StreamingResponse(
            output,
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename={report_type}_report.csv"}
        )
    else:
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='Report')
        output.seek(0)
        return StreamingResponse(
            output,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition": f"attachment; filename={report_type}_report.xlsx"}
        )

# Include the router in the main app
app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get('CORS_ORIGINS', '*').split(','),
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()

import uvicorn

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=port,
        reload=False
    )