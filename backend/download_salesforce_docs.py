#!/usr/bin/env python3
"""
Download Official Salesforce Documentation PDFs
Sources: resources.docs.salesforce.com (Official Salesforce)
"""

import os
import requests
from urllib.parse import urlparse
from pathlib import Path

# Directory to store downloaded PDFs
DOCS_DIR = Path(__file__).parent / "salesforce_docs"
DOCS_DIR.mkdir(exist_ok=True)

# Official Salesforce Documentation PDFs
SALESFORCE_PDFS = {
    # ============ CORE ADMIN & BASICS ============
    "basics": {
        "url": "https://resources.docs.salesforce.com/latest/latest/en-us/sfdc/pdf/basics.pdf",
        "category": "admin",
        "description": "Get Started with Salesforce - Core Admin Guide"
    },
    "setup": {
        "url": "https://resources.docs.salesforce.com/latest/latest/en-us/sfdc/pdf/setup_overview.pdf",
        "category": "admin",
        "description": "Salesforce Setup Overview"
    },

    # ============ SALES CLOUD / CRM ============
    "sales_core": {
        "url": "https://resources.docs.salesforce.com/latest/latest/en-us/sfdc/pdf/sales_core.pdf",
        "category": "sales",
        "description": "Sales Cloud Core Features"
    },
    "leads_accounts": {
        "url": "https://resources.docs.salesforce.com/latest/latest/en-us/sfdc/pdf/sales_leads.pdf",
        "category": "sales",
        "description": "Leads and Accounts Management"
    },
    "opportunities": {
        "url": "https://resources.docs.salesforce.com/latest/latest/en-us/sfdc/pdf/sales_opportunities.pdf",
        "category": "sales",
        "description": "Opportunities and Pipeline Management"
    },

    # ============ MARKETING CLOUD ============
    "marketing_implementation": {
        "url": "https://resources.docs.salesforce.com/latest/latest/en-us/sfdc/pdf/marketing_setup_implementation_guide.pdf",
        "category": "marketing",
        "description": "Marketing Cloud Implementation Guide"
    },
    "marketing_email": {
        "url": "https://resources.docs.salesforce.com/latest/latest/en-us/sfdc/pdf/mc_email.pdf",
        "category": "marketing",
        "description": "Marketing Cloud Email Guide"
    },
    "marketing_journeys": {
        "url": "https://resources.docs.salesforce.com/latest/latest/en-us/sfdc/pdf/mc_journeys_and_automations.pdf",
        "category": "marketing",
        "description": "Marketing Cloud Journeys and Automations"
    },
    "pardot_implementation": {
        "url": "https://resources.docs.salesforce.com/latest/latest/en-us/sfdc/pdf/pardot_implementation_guide.pdf",
        "category": "marketing",
        "description": "Account Engagement (Pardot) Implementation Guide"
    },

    # ============ SERVICE CLOUD ============
    "service_cloud": {
        "url": "https://resources.docs.salesforce.com/latest/latest/en-us/sfdc/pdf/service_presence.pdf",
        "category": "service",
        "description": "Service Cloud Presence and Routing"
    },
    "case_management": {
        "url": "https://resources.docs.salesforce.com/latest/latest/en-us/sfdc/pdf/cases_def.pdf",
        "category": "service",
        "description": "Case Management Guide"
    },

    # ============ AUTOMATION & FLOWS ============
    "flow_builder": {
        "url": "https://resources.docs.salesforce.com/latest/latest/en-us/sfdc/pdf/salesforce_flow.pdf",
        "category": "automation",
        "description": "Flow Builder Complete Guide"
    },
    "process_automation": {
        "url": "https://resources.docs.salesforce.com/latest/latest/en-us/sfdc/pdf/process_automation.pdf",
        "category": "automation",
        "description": "Process Automation Guide"
    },

    # ============ DEVELOPMENT ============
    "apex_dev": {
        "url": "https://resources.docs.salesforce.com/latest/latest/en-us/sfdc/pdf/apex_developer_guide.pdf",
        "category": "development",
        "description": "Apex Developer Guide"
    },
    "soql_sosl": {
        "url": "https://resources.docs.salesforce.com/latest/latest/en-us/sfdc/pdf/salesforce_soql_sosl.pdf",
        "category": "development",
        "description": "SOQL and SOSL Reference"
    },
    "lightning_components": {
        "url": "https://resources.docs.salesforce.com/latest/latest/en-us/sfdc/pdf/lightning.pdf",
        "category": "development",
        "description": "Lightning Components Developer Guide"
    },
    "rest_api": {
        "url": "https://resources.docs.salesforce.com/latest/latest/en-us/sfdc/pdf/api_rest.pdf",
        "category": "development",
        "description": "REST API Developer Guide"
    },

    # ============ SECURITY & DATA ============
    "security_guide": {
        "url": "https://resources.docs.salesforce.com/latest/latest/en-us/sfdc/pdf/salesforce_security_guide.pdf",
        "category": "security",
        "description": "Salesforce Security Guide"
    },
    "sharing_settings": {
        "url": "https://resources.docs.salesforce.com/latest/latest/en-us/sfdc/pdf/sharing.pdf",
        "category": "security",
        "description": "Sharing and Visibility Guide"
    },
    "data_loader": {
        "url": "https://resources.docs.salesforce.com/latest/latest/en-us/sfdc/pdf/data_loader.pdf",
        "category": "data",
        "description": "Data Loader Guide"
    },

    # ============ REPORTS & ANALYTICS ============
    "reports_dashboards": {
        "url": "https://resources.docs.salesforce.com/latest/latest/en-us/sfdc/pdf/salesforce_reports_dashboards.pdf",
        "category": "analytics",
        "description": "Reports and Dashboards Guide"
    },

    # ============ CERTIFICATION GUIDES ============
    "cert_admin": {
        "url": "https://developer.salesforce.com/resources2/certification-site/files/SGCertifiedAdministrator.pdf",
        "category": "certification",
        "description": "Certified Administrator Exam Guide"
    },
    "cert_advanced_admin": {
        "url": "https://developer.salesforce.com/resources2/certification-site/files/SGCertifiedAdvancedAdministrator.pdf",
        "category": "certification",
        "description": "Certified Advanced Administrator Exam Guide"
    },
    "cert_marketing_cloud": {
        "url": "https://developer.salesforce.com/resources2/certification-site/files/SGCertifiedMarketingCloudConsultant.pdf",
        "category": "certification",
        "description": "Certified Marketing Cloud Consultant Exam Guide"
    },
}

def download_pdf(name: str, info: dict) -> bool:
    """Download a single PDF file."""
    url = info["url"]
    filename = f"{info['category']}_{name}.pdf"
    filepath = DOCS_DIR / filename

    if filepath.exists():
        print(f"  ✓ Already exists: {filename}")
        return True

    try:
        print(f"  ↓ Downloading: {filename}...")
        response = requests.get(url, timeout=60, headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        })

        if response.status_code == 200:
            # Check if it's actually a PDF
            content_type = response.headers.get("content-type", "")
            if "pdf" in content_type.lower() or response.content[:4] == b"%PDF":
                with open(filepath, "wb") as f:
                    f.write(response.content)
                size_mb = len(response.content) / (1024 * 1024)
                print(f"    ✓ Saved: {filename} ({size_mb:.1f} MB)")
                return True
            else:
                print(f"    ✗ Not a PDF: {filename} (got {content_type})")
                return False
        else:
            print(f"    ✗ HTTP {response.status_code}: {filename}")
            return False

    except Exception as e:
        print(f"    ✗ Error: {filename} - {e}")
        return False

def main():
    print("=" * 60)
    print("SALESFORCE DOCUMENTATION DOWNLOADER")
    print("=" * 60)
    print(f"\nDownload directory: {DOCS_DIR}")
    print(f"Total PDFs to download: {len(SALESFORCE_PDFS)}\n")

    # Group by category
    categories = {}
    for name, info in SALESFORCE_PDFS.items():
        cat = info["category"]
        if cat not in categories:
            categories[cat] = []
        categories[cat].append((name, info))

    success = 0
    failed = 0

    for category, docs in categories.items():
        print(f"\n[{category.upper()}] ({len(docs)} docs)")
        print("-" * 40)

        for name, info in docs:
            if download_pdf(name, info):
                success += 1
            else:
                failed += 1

    print("\n" + "=" * 60)
    print(f"DOWNLOAD COMPLETE: {success} success, {failed} failed")
    print("=" * 60)

    # List downloaded files
    pdfs = list(DOCS_DIR.glob("*.pdf"))
    if pdfs:
        total_size = sum(p.stat().st_size for p in pdfs) / (1024 * 1024)
        print(f"\nDownloaded {len(pdfs)} PDFs ({total_size:.1f} MB total)")
        print("\nNext step: Run 'python ingest_salesforce_docs.py' to ingest into RAG")

if __name__ == "__main__":
    main()
