"""
Salesforce API Service
Using OAuth Client Credentials flow for the Virtual Consultant
"""
import os
import logging
import requests
from typing import Dict, List, Optional, Any
from simple_salesforce import Salesforce

logger = logging.getLogger(__name__)

# Salesforce OAuth Credentials (set in .env file)
SF_CLIENT_ID = os.getenv("SALESFORCE_CLIENT_ID", "")
SF_CLIENT_SECRET = os.getenv("SALESFORCE_CLIENT_SECRET", "")
SF_LOGIN_URL = os.getenv("SALESFORCE_LOGIN_URL", "https://login.salesforce.com")


class SalesforceService:
    """Service for interacting with Salesforce REST API"""

    def __init__(self):
        self._sf: Optional[Salesforce] = None
        self._access_token: Optional[str] = None
        self._instance_url: Optional[str] = None

    def _get_oauth_token(self) -> tuple:
        """Get OAuth token using Client Credentials flow"""
        token_url = f"{SF_LOGIN_URL}/services/oauth2/token"

        data = {
            "grant_type": "client_credentials",
            "client_id": SF_CLIENT_ID,
            "client_secret": SF_CLIENT_SECRET
        }

        response = requests.post(token_url, data=data, timeout=30)

        if response.status_code != 200:
            error_msg = response.json().get("error_description", response.text)
            raise Exception(f"OAuth failed: {error_msg}")

        result = response.json()
        return result["access_token"], result["instance_url"]

    def connect(self) -> Salesforce:
        """Connect to Salesforce using OAuth"""
        if self._sf:
            return self._sf

        try:
            # Get OAuth token
            access_token, instance_url = self._get_oauth_token()
            self._access_token = access_token
            self._instance_url = instance_url

            # Create Salesforce connection with token
            self._sf = Salesforce(
                instance_url=instance_url,
                session_id=access_token
            )
            logger.info(f"Connected to Salesforce: {instance_url}")
            return self._sf
        except Exception as e:
            logger.error(f"Salesforce connection failed: {e}")
            raise Exception(f"Failed to connect: {e}")

    def query(self, soql: str) -> Dict[str, Any]:
        """Execute SOQL query"""
        sf = self.connect()
        try:
            result = sf.query_all(soql)
            logger.info(f"Query returned {result.get('totalSize', 0)} records")
            return dict(result)
        except Exception as e:
            logger.error(f"Query failed: {e}")
            raise Exception(f"SOQL query failed: {e}")

    def search(self, sosl: str) -> Dict[str, Any]:
        """Execute SOSL search"""
        sf = self.connect()
        try:
            result = sf.search(sosl)
            return dict(result) if result else {"searchRecords": []}
        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise Exception(f"SOSL search failed: {e}")

    def describe_object(self, object_name: str) -> Dict[str, Any]:
        """Get object metadata"""
        sf = self.connect()
        try:
            sobject = getattr(sf, object_name)
            result = sobject.describe()
            return dict(result)
        except Exception as e:
            logger.error(f"Describe failed: {e}")
            raise Exception(f"Describe object failed: {e}")

    def describe_global(self) -> Dict[str, Any]:
        """List all objects"""
        sf = self.connect()
        try:
            result = sf.describe()
            return dict(result)
        except Exception as e:
            logger.error(f"Describe global failed: {e}")
            raise Exception(f"Describe global failed: {e}")

    def create_record(self, object_name: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new record"""
        sf = self.connect()
        try:
            sobject = getattr(sf, object_name)
            result = sobject.create(data)
            logger.info(f"Created {object_name} record: {result.get('id')}")
            return dict(result)
        except Exception as e:
            logger.error(f"Create failed: {e}")
            raise Exception(f"Create record failed: {e}")

    def update_record(self, object_name: str, record_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Update an existing record"""
        sf = self.connect()
        try:
            sobject = getattr(sf, object_name)
            result = sobject.update(record_id, data)
            logger.info(f"Updated {object_name} record: {record_id}")
            return {"success": True, "id": record_id}
        except Exception as e:
            logger.error(f"Update failed: {e}")
            raise Exception(f"Update record failed: {e}")

    def delete_record(self, object_name: str, record_id: str) -> Dict[str, Any]:
        """Delete a record"""
        sf = self.connect()
        try:
            sobject = getattr(sf, object_name)
            result = sobject.delete(record_id)
            logger.info(f"Deleted {object_name} record: {record_id}")
            return {"success": True, "id": record_id}
        except Exception as e:
            logger.error(f"Delete failed: {e}")
            raise Exception(f"Delete record failed: {e}")

    def execute_apex(self, code: str) -> Dict[str, Any]:
        """Execute anonymous Apex"""
        sf = self.connect()
        try:
            result = sf.restful(
                f"tooling/executeAnonymous?anonymousBody={code}",
                method="GET"
            )
            return dict(result) if result else {"success": True}
        except Exception as e:
            logger.error(f"Apex execution failed: {e}")
            raise Exception(f"Apex execution failed: {e}")

    def get_org_info(self) -> Dict[str, Any]:
        """Get organization information"""
        try:
            sf = self.connect()
            result = sf.query("SELECT Id, Name, OrganizationType, IsSandbox FROM Organization LIMIT 1")
            if result.get("records"):
                org = result["records"][0]
                return {
                    "connected": True,
                    "org_id": org.get("Id"),
                    "org_name": org.get("Name"),
                    "org_type": org.get("OrganizationType"),
                    "is_sandbox": org.get("IsSandbox", False),
                    "instance_url": f"https://{sf.sf_instance}",
                    "username": "OAuth Client Credentials"
                }
            return {"connected": False}
        except Exception as e:
            logger.error(f"Get org info failed: {e}")
            return {"connected": False, "error": str(e)}

    def get_recent_records(self, object_name: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent records for an object"""
        try:
            result = self.query(f"SELECT Id, Name, CreatedDate FROM {object_name} ORDER BY CreatedDate DESC LIMIT {limit}")
            return result.get("records", [])
        except Exception as e:
            logger.error(f"Get recent records failed: {e}")
            return []


# Singleton instance
_salesforce_service: Optional[SalesforceService] = None


def get_salesforce_service() -> SalesforceService:
    """Get or create Salesforce service singleton"""
    global _salesforce_service
    if _salesforce_service is None:
        _salesforce_service = SalesforceService()
    return _salesforce_service


# Test function
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("Testing Salesforce connection...")
    service = get_salesforce_service()

    try:
        # Test connection
        sf = service.connect()
        print(f"✓ Connected to: {sf.sf_instance}")

        # Test org info
        org_info = service.get_org_info()
        print(f"✓ Org: {org_info.get('org_name')} ({org_info.get('org_type')})")

        # Test query
        accounts = service.query("SELECT Id, Name FROM Account LIMIT 5")
        print(f"✓ Found {accounts.get('totalSize', 0)} accounts")
        for acc in accounts.get("records", [])[:3]:
            print(f"  - {acc.get('Name')}")

        # Test describe global
        global_desc = service.describe_global()
        sobject_count = len(global_desc.get("sobjects", []))
        print(f"✓ Org has {sobject_count} objects")

        # Test describe Account
        account_desc = service.describe_object("Account")
        field_count = len(account_desc.get("fields", []))
        print(f"✓ Account has {field_count} fields")

        print("\n✅ All Salesforce tests passed!")
        print(f"\nOrg Details:")
        print(f"  Name: {org_info.get('org_name')}")
        print(f"  Type: {org_info.get('org_type')}")
        print(f"  Instance: {org_info.get('instance_url')}")

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
