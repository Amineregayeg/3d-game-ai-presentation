"""
Salesforce MCP API Blueprint
Provides REST endpoints for Salesforce operations
"""
import logging
from flask import Blueprint, request, jsonify
from salesforce_service import get_salesforce_service

logger = logging.getLogger(__name__)

salesforce_mcp_bp = Blueprint('salesforce_mcp', __name__, url_prefix='/api/salesforce-mcp')


@salesforce_mcp_bp.route('/status', methods=['GET'])
def get_status():
    """Get Salesforce connection status and org info"""
    try:
        service = get_salesforce_service()
        org_info = service.get_org_info()
        return jsonify(org_info)
    except Exception as e:
        logger.error(f"Status check failed: {e}")
        return jsonify({"connected": False, "error": str(e)}), 500


@salesforce_mcp_bp.route('/query', methods=['POST'])
def execute_query():
    """Execute SOQL query"""
    try:
        data = request.json
        soql = data.get('soql')

        if not soql:
            return jsonify({"error": "soql parameter required"}), 400

        service = get_salesforce_service()
        result = service.query(soql)

        return jsonify({
            "success": True,
            "totalSize": result.get("totalSize", 0),
            "records": result.get("records", []),
            "done": result.get("done", True)
        })
    except Exception as e:
        logger.error(f"Query failed: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@salesforce_mcp_bp.route('/describe/<object_name>', methods=['GET'])
def describe_object(object_name):
    """Get object metadata"""
    try:
        service = get_salesforce_service()
        result = service.describe_object(object_name)

        # Return simplified field info
        fields = [{
            "name": f.get("name"),
            "label": f.get("label"),
            "type": f.get("type"),
            "required": not f.get("nillable", True),
            "updateable": f.get("updateable", False)
        } for f in result.get("fields", [])]

        return jsonify({
            "success": True,
            "name": result.get("name"),
            "label": result.get("label"),
            "fieldCount": len(fields),
            "fields": fields
        })
    except Exception as e:
        logger.error(f"Describe failed: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@salesforce_mcp_bp.route('/objects', methods=['GET'])
def list_objects():
    """List all Salesforce objects"""
    try:
        service = get_salesforce_service()
        result = service.describe_global()

        # Return simplified object list
        objects = [{
            "name": obj.get("name"),
            "label": obj.get("label"),
            "custom": obj.get("custom", False),
            "queryable": obj.get("queryable", False)
        } for obj in result.get("sobjects", [])]

        # Filter to queryable objects only
        queryable_objects = [o for o in objects if o["queryable"]]

        return jsonify({
            "success": True,
            "totalCount": len(objects),
            "queryableCount": len(queryable_objects),
            "objects": queryable_objects[:100]  # Limit to 100
        })
    except Exception as e:
        logger.error(f"List objects failed: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@salesforce_mcp_bp.route('/create', methods=['POST'])
def create_record():
    """Create a new record"""
    try:
        data = request.json
        object_name = data.get('objectName')
        record_data = data.get('data')

        if not object_name or not record_data:
            return jsonify({"error": "objectName and data required"}), 400

        service = get_salesforce_service()
        result = service.create_record(object_name, record_data)

        return jsonify({
            "success": True,
            "id": result.get("id"),
            "message": f"Created {object_name} record successfully"
        })
    except Exception as e:
        logger.error(f"Create failed: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@salesforce_mcp_bp.route('/update', methods=['POST'])
def update_record():
    """Update an existing record"""
    try:
        data = request.json
        object_name = data.get('objectName')
        record_id = data.get('recordId')
        record_data = data.get('data')

        if not object_name or not record_id or not record_data:
            return jsonify({"error": "objectName, recordId, and data required"}), 400

        service = get_salesforce_service()
        result = service.update_record(object_name, record_id, record_data)

        return jsonify({
            "success": True,
            "id": record_id,
            "message": f"Updated {object_name} record successfully"
        })
    except Exception as e:
        logger.error(f"Update failed: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@salesforce_mcp_bp.route('/delete', methods=['POST'])
def delete_record():
    """Delete a record"""
    try:
        data = request.json
        object_name = data.get('objectName')
        record_id = data.get('recordId')

        if not object_name or not record_id:
            return jsonify({"error": "objectName and recordId required"}), 400

        service = get_salesforce_service()
        result = service.delete_record(object_name, record_id)

        return jsonify({
            "success": True,
            "id": record_id,
            "message": f"Deleted {object_name} record successfully"
        })
    except Exception as e:
        logger.error(f"Delete failed: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@salesforce_mcp_bp.route('/execute', methods=['POST'])
def execute_operation():
    """Execute a Salesforce MCP operation (unified endpoint)"""
    try:
        data = request.json
        operation = data.get('operation')
        params = data.get('params', {})

        service = get_salesforce_service()

        if operation == 'query':
            result = service.query(params.get('soql', ''))
            return jsonify({
                "success": True,
                "operation": "query",
                "result": {
                    "totalSize": result.get("totalSize", 0),
                    "records": result.get("records", [])
                }
            })

        elif operation == 'describe':
            result = service.describe_object(params.get('objectName', 'Account'))
            return jsonify({
                "success": True,
                "operation": "describe",
                "result": {
                    "name": result.get("name"),
                    "label": result.get("label"),
                    "fieldCount": len(result.get("fields", []))
                }
            })

        elif operation == 'create':
            result = service.create_record(
                params.get('objectName'),
                params.get('data', {})
            )
            return jsonify({
                "success": True,
                "operation": "create",
                "result": {"id": result.get("id")}
            })

        elif operation == 'update':
            result = service.update_record(
                params.get('objectName'),
                params.get('recordId'),
                params.get('data', {})
            )
            return jsonify({
                "success": True,
                "operation": "update",
                "result": result
            })

        elif operation == 'delete':
            result = service.delete_record(
                params.get('objectName'),
                params.get('recordId')
            )
            return jsonify({
                "success": True,
                "operation": "delete",
                "result": result
            })

        else:
            return jsonify({
                "success": False,
                "error": f"Unknown operation: {operation}"
            }), 400

    except Exception as e:
        logger.error(f"Execute operation failed: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@salesforce_mcp_bp.route('/limits', methods=['GET'])
def get_org_limits():
    """Get org limits and usage statistics"""
    try:
        service = get_salesforce_service()
        sf = service.connect()

        # Get org limits
        limits = sf.restful('limits/')

        # Get key limits
        key_limits = {
            "DailyApiRequests": limits.get("DailyApiRequests", {}),
            "DailyBulkApiRequests": limits.get("DailyBulkApiRequests", {}),
            "DailyAsyncApexExecutions": limits.get("DailyAsyncApexExecutions", {}),
            "DataStorageMB": limits.get("DataStorageMB", {}),
            "FileStorageMB": limits.get("FileStorageMB", {}),
            "DailyWorkflowEmails": limits.get("DailyWorkflowEmails", {}),
            "HourlyTimeBasedWorkflow": limits.get("HourlyTimeBasedWorkflow", {}),
            "DailyDurableStreamingApiEvents": limits.get("DailyDurableStreamingApiEvents", {}),
            "StreamingApiConcurrentClients": limits.get("StreamingApiConcurrentClients", {}),
            "SingleEmail": limits.get("SingleEmail", {}),
            "MassEmail": limits.get("MassEmail", {}),
        }

        return jsonify({
            "success": True,
            "limits": key_limits,
            "allLimits": limits
        })
    except Exception as e:
        logger.error(f"Get limits failed: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@salesforce_mcp_bp.route('/dashboard', methods=['GET'])
def get_org_dashboard():
    """Get comprehensive org dashboard data"""
    try:
        service = get_salesforce_service()
        sf = service.connect()

        # Get org info
        org_result = sf.query("SELECT Id, Name, OrganizationType, IsSandbox, InstanceName, LanguageLocaleKey FROM Organization LIMIT 1")
        org_info = org_result.get("records", [{}])[0] if org_result.get("records") else {}

        # Get user count
        user_result = sf.query("SELECT COUNT(Id) total FROM User WHERE IsActive = true")
        active_users = user_result.get("records", [{}])[0].get("total", 0) if user_result.get("records") else 0

        # Get record counts for key objects
        counts = {}
        for obj in ["Account", "Contact", "Lead", "Opportunity", "Case"]:
            try:
                result = sf.query(f"SELECT COUNT(Id) total FROM {obj}")
                counts[obj] = result.get("records", [{}])[0].get("total", 0) if result.get("records") else 0
            except:
                counts[obj] = 0

        # Get limits
        limits = sf.restful('limits/')

        # Get recent items (last 5 of each type)
        recent_accounts = []
        recent_contacts = []
        try:
            acc_result = sf.query("SELECT Id, Name, Industry, CreatedDate FROM Account ORDER BY CreatedDate DESC LIMIT 5")
            recent_accounts = acc_result.get("records", [])
        except:
            pass
        try:
            con_result = sf.query("SELECT Id, FirstName, LastName, Email, CreatedDate FROM Contact ORDER BY CreatedDate DESC LIMIT 5")
            recent_contacts = con_result.get("records", [])
        except:
            pass

        return jsonify({
            "success": True,
            "org": {
                "id": org_info.get("Id"),
                "name": org_info.get("Name"),
                "type": org_info.get("OrganizationType"),
                "isSandbox": org_info.get("IsSandbox", False),
                "instance": org_info.get("InstanceName"),
                "locale": org_info.get("LanguageLocaleKey"),
                "instanceUrl": f"https://{sf.sf_instance}"
            },
            "stats": {
                "activeUsers": active_users,
                "recordCounts": counts
            },
            "limits": {
                "apiRequests": limits.get("DailyApiRequests", {}),
                "dataStorage": limits.get("DataStorageMB", {}),
                "fileStorage": limits.get("FileStorageMB", {})
            },
            "recent": {
                "accounts": recent_accounts,
                "contacts": recent_contacts
            }
        })
    except Exception as e:
        logger.error(f"Get dashboard failed: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@salesforce_mcp_bp.route('/describe/<object_name>/full', methods=['GET'])
def describe_object_full(object_name):
    """Get full object metadata including relationships"""
    try:
        service = get_salesforce_service()
        result = service.describe_object(object_name)

        # Organize fields by category
        fields = []
        relationships = []

        for f in result.get("fields", []):
            field_info = {
                "name": f.get("name"),
                "label": f.get("label"),
                "type": f.get("type"),
                "length": f.get("length"),
                "precision": f.get("precision"),
                "scale": f.get("scale"),
                "required": not f.get("nillable", True),
                "unique": f.get("unique", False),
                "updateable": f.get("updateable", False),
                "createable": f.get("createable", False),
                "custom": f.get("custom", False),
                "defaultValue": f.get("defaultValue"),
                "picklistValues": [
                    {"value": p.get("value"), "label": p.get("label"), "active": p.get("active")}
                    for p in f.get("picklistValues", [])
                ] if f.get("type") == "picklist" or f.get("type") == "multipicklist" else None,
                "referenceTo": f.get("referenceTo", []),
                "relationshipName": f.get("relationshipName")
            }

            if f.get("type") == "reference" and f.get("referenceTo"):
                relationships.append(field_info)
            else:
                fields.append(field_info)

        # Get child relationships
        child_relationships = [
            {
                "name": r.get("relationshipName"),
                "childObject": r.get("childSObject"),
                "field": r.get("field"),
                "cascadeDelete": r.get("cascadeDelete", False)
            }
            for r in result.get("childRelationships", [])
            if r.get("relationshipName")
        ]

        return jsonify({
            "success": True,
            "object": {
                "name": result.get("name"),
                "label": result.get("label"),
                "labelPlural": result.get("labelPlural"),
                "keyPrefix": result.get("keyPrefix"),
                "custom": result.get("custom", False),
                "createable": result.get("createable", False),
                "updateable": result.get("updateable", False),
                "deletable": result.get("deletable", False),
                "queryable": result.get("queryable", False),
                "searchable": result.get("searchable", False)
            },
            "fields": fields,
            "relationships": relationships,
            "childRelationships": child_relationships[:20],  # Limit to 20
            "fieldCount": len(fields),
            "relationshipCount": len(relationships)
        })
    except Exception as e:
        logger.error(f"Describe full failed: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@salesforce_mcp_bp.route('/record/<object_name>/<record_id>', methods=['GET'])
def get_record(object_name, record_id):
    """Get full record details"""
    try:
        service = get_salesforce_service()
        sf = service.connect()

        # First describe to get all fields
        sobject = getattr(sf, object_name)
        describe = sobject.describe()

        # Get queryable field names
        field_names = [f.get("name") for f in describe.get("fields", []) if f.get("type") not in ["address", "location"]]

        # Query the record with all fields
        fields_str = ", ".join(field_names[:50])  # Limit to 50 fields
        result = sf.query(f"SELECT {fields_str} FROM {object_name} WHERE Id = '{record_id}' LIMIT 1")

        if not result.get("records"):
            return jsonify({"success": False, "error": "Record not found"}), 404

        record = result.get("records")[0]

        # Build field display data
        field_data = []
        for f in describe.get("fields", []):
            name = f.get("name")
            if name in record:
                field_data.append({
                    "name": name,
                    "label": f.get("label"),
                    "type": f.get("type"),
                    "value": record.get(name)
                })

        return jsonify({
            "success": True,
            "object": object_name,
            "id": record_id,
            "record": record,
            "fields": field_data
        })
    except Exception as e:
        logger.error(f"Get record failed: {e}")
        return jsonify({"success": False, "error": str(e)}), 500
