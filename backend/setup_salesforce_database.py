"""
Salesforce RAG Database Setup Script
Creates PostgreSQL schema with pgvector support for Salesforce documentation
"""

import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
import os

# Database configuration (same DB, different table)
DB_CONFIG = {
    "host": os.environ.get("SALESFORCE_DB_HOST", "localhost"),
    "port": int(os.environ.get("SALESFORCE_DB_PORT", 5432)),
    "dbname": os.environ.get("SALESFORCE_DB_NAME", "ragdb"),
    "user": os.environ.get("SALESFORCE_DB_USER", "raguser"),
    "password": os.environ.get("SALESFORCE_DB_PASSWORD", "<RAG_DB_PASSWORD>")
}

EMBEDDING_DIM = 384  # all-MiniLM-L6-v2 for fast CPU inference


def create_salesforce_schema():
    """Create database schema for Salesforce RAG system"""

    conn = psycopg2.connect(**DB_CONFIG)
    conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    cur = conn.cursor()

    print("Creating Salesforce RAG database schema...")

    # Enable pgvector extension
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    print("✓ pgvector extension enabled")

    # Create salesforce_documents table
    cur.execute(f"""
        CREATE TABLE IF NOT EXISTS salesforce_documents (
            id BIGSERIAL PRIMARY KEY,

            -- Content
            content TEXT NOT NULL,
            title VARCHAR(500),
            embedding vector({EMBEDDING_DIM}),

            -- Salesforce-specific metadata
            source VARCHAR(50) DEFAULT 'help_docs',
            salesforce_version VARCHAR(20) DEFAULT 'Winter 25',
            category VARCHAR(50) DEFAULT 'general',
            subcategory VARCHAR(100),

            -- Content classification
            object_types TEXT[],
            features TEXT[],
            api_names TEXT[],

            -- Content type
            language VARCHAR(20) DEFAULT 'english',
            is_code BOOLEAN DEFAULT FALSE,
            is_apex BOOLEAN DEFAULT FALSE,
            is_soql BOOLEAN DEFAULT FALSE,
            is_flow BOOLEAN DEFAULT FALSE,
            difficulty VARCHAR(20) DEFAULT 'intermediate',

            -- Ranking
            priority FLOAT DEFAULT 0.5,
            relevance_score FLOAT DEFAULT 0.0,

            -- Temporal
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

            -- Source tracking
            source_url VARCHAR(1000),
            source_hash VARCHAR(64),
            trailhead_module VARCHAR(200),

            -- Tags
            tags TEXT[],
            keywords TEXT[],
            certifications TEXT[],

            -- Full-text search
            tsv tsvector GENERATED ALWAYS AS (
                to_tsvector('english', content || ' ' || COALESCE(title, ''))
            ) STORED
        );
    """)
    print("✓ salesforce_documents table created")

    # Create HNSW index for vector search
    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_sf_documents_embedding
        ON salesforce_documents
        USING hnsw (embedding vector_cosine_ops)
        WITH (m = 16, ef_construction = 200);
    """)
    print("✓ HNSW vector index created")

    # Create GIN index for full-text search
    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_sf_documents_tsv
        ON salesforce_documents
        USING GIN (tsv);
    """)
    print("✓ GIN full-text index created")

    # Create other useful indexes
    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_sf_documents_source_version
        ON salesforce_documents(source, salesforce_version);
    """)
    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_sf_documents_category
        ON salesforce_documents(category, subcategory);
    """)
    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_sf_documents_object_types
        ON salesforce_documents USING GIN (object_types);
    """)
    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_sf_documents_features
        ON salesforce_documents USING GIN (features);
    """)
    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_sf_documents_tags
        ON salesforce_documents USING GIN (tags);
    """)
    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_sf_documents_is_apex
        ON salesforce_documents(is_apex);
    """)
    print("✓ Additional indexes created")

    # Create Salesforce-specific sessions table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS salesforce_sessions (
            id VARCHAR(100) PRIMARY KEY,
            org_id VARCHAR(50),
            user_id VARCHAR(50),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            history JSONB DEFAULT '[]'::jsonb,
            context JSONB DEFAULT '{}'::jsonb,
            mcp_operations JSONB DEFAULT '[]'::jsonb
        );
    """)
    print("✓ salesforce_sessions table created")

    # Create retrieval history for Salesforce
    cur.execute(f"""
        CREATE TABLE IF NOT EXISTS salesforce_retrieval_history (
            id BIGSERIAL PRIMARY KEY,
            query TEXT NOT NULL,
            query_embedding vector({EMBEDDING_DIM}),
            intent VARCHAR(100),
            retrieved_doc_ids BIGINT[],
            final_answer TEXT,
            mcp_operations_executed TEXT[],
            user_satisfaction FLOAT,
            latency_ms INTEGER,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            session_id VARCHAR(100)
        );
    """)
    print("✓ salesforce_retrieval_history table created")

    cur.close()
    conn.close()

    print("\n✅ Salesforce database schema created successfully!")


def insert_sample_salesforce_documents():
    """Insert sample Salesforce documentation for testing"""

    from sentence_transformers import SentenceTransformer

    print("\nLoading embedding model...")
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    print(f"✓ Model loaded (dimension: {model.get_sentence_embedding_dimension()})")

    # Sample Salesforce documentation - comprehensive knowledge base
    sample_docs = [
        # === ACCOUNT MANAGEMENT ===
        {
            "title": "Creating Custom Fields on Account Object",
            "content": """To create a custom field on the Account object in Salesforce:

1. Go to Setup (gear icon)
2. Navigate to Object Manager > Account
3. Click on "Fields & Relationships"
4. Click "New" button
5. Select the field type (Text, Number, Picklist, etc.)
6. Configure field properties:
   - Field Label: User-friendly name
   - Field Name: API name (auto-generated)
   - Length: For text fields
   - Required: Make mandatory if needed
7. Set field-level security
8. Add to page layouts
9. Click "Save"

Best Practice: Use descriptive API names and always set appropriate field-level security before deploying to production.

Apex Example:
```apex
// Query custom field
List<Account> accounts = [SELECT Id, Name, Custom_Field__c FROM Account LIMIT 10];

// Update custom field
Account acc = new Account(Id = '001xx000003DGb2', Custom_Field__c = 'New Value');
update acc;
```""",
            "source": "help_docs",
            "category": "configuration",
            "subcategory": "custom_fields",
            "object_types": ["Account"],
            "features": ["Custom Fields", "Object Manager"],
            "is_code": True,
            "is_apex": True,
            "difficulty": "beginner",
            "tags": ["custom fields", "account", "configuration", "apex"],
            "keywords": ["field", "custom", "account", "setup", "object manager"]
        },
        {
            "title": "Account Hierarchy and Parent-Child Relationships",
            "content": """Salesforce Account Hierarchy allows you to create parent-child relationships between accounts.

Key Concepts:
- Parent Account: The main/holding company
- Child Account: Subsidiary or branch of parent
- Hierarchy Path: Visual representation of relationships

Setting Up Account Hierarchy:
1. Open the child account record
2. Find the "Parent Account" field
3. Look up and select the parent account
4. Save the record

Viewing Hierarchy:
1. Go to any account in the hierarchy
2. Click "View Account Hierarchy" in the highlights panel
3. Navigate up/down the hierarchy tree

SOQL for Hierarchy:
```soql
-- Get account with parent info
SELECT Id, Name, Parent.Name, Parent.Id
FROM Account
WHERE ParentId != null

-- Get all child accounts
SELECT Id, Name
FROM Account
WHERE ParentId = '001xxxxxxxxxxxx'

-- Traverse hierarchy (up to 5 levels)
SELECT Id, Name, Parent.Parent.Parent.Name
FROM Account
```

Best Practice: Limit hierarchy depth to 5 levels for query performance. Use "Ultimate Parent" formula fields for deep hierarchies.""",
            "source": "help_docs",
            "category": "configuration",
            "subcategory": "relationships",
            "object_types": ["Account"],
            "features": ["Account Hierarchy", "Relationships"],
            "is_code": True,
            "is_soql": True,
            "difficulty": "intermediate",
            "tags": ["hierarchy", "parent", "child", "relationships", "soql"],
            "keywords": ["parent account", "hierarchy", "child account", "relationship"]
        },

        # === OPPORTUNITY MANAGEMENT ===
        {
            "title": "Opportunity Stages and Sales Path",
            "content": """Salesforce Opportunity Stages track the sales process from lead to closed deal.

Default Opportunity Stages:
1. Prospecting (10%)
2. Qualification (20%)
3. Needs Analysis (30%)
4. Value Proposition (50%)
5. Id. Decision Makers (60%)
6. Perception Analysis (70%)
7. Proposal/Price Quote (75%)
8. Negotiation/Review (90%)
9. Closed Won (100%)
10. Closed Lost (0%)

Customizing Stages:
1. Setup > Object Manager > Opportunity
2. Fields & Relationships > Stage
3. Edit picklist values
4. Adjust probability percentages

Sales Path Setup:
1. Setup > Sales Path
2. Create new Sales Path for Opportunity
3. Add guidance, fields, and resources per stage
4. Activate the path

Apex for Stage Updates:
```apex
// Update opportunity stage
Opportunity opp = [SELECT Id, StageName FROM Opportunity WHERE Id = :oppId];
opp.StageName = 'Proposal/Price Quote';
opp.Probability = 75;
update opp;

// Trigger on stage change
trigger OpportunityTrigger on Opportunity (after update) {
    for (Opportunity opp : Trigger.new) {
        if (opp.StageName != Trigger.oldMap.get(opp.Id).StageName) {
            // Stage changed - take action
        }
    }
}
```""",
            "source": "help_docs",
            "category": "sales",
            "subcategory": "opportunities",
            "object_types": ["Opportunity"],
            "features": ["Opportunity Stages", "Sales Path"],
            "is_code": True,
            "is_apex": True,
            "difficulty": "intermediate",
            "tags": ["opportunity", "stages", "sales path", "pipeline", "apex"],
            "keywords": ["opportunity", "stage", "sales", "pipeline", "probability"]
        },
        {
            "title": "Opportunity Products and Price Books",
            "content": """Opportunity Products (Line Items) connect Products and Price Books to Opportunities.

Key Objects:
- Product2: Master product catalog
- PricebookEntry: Product price in a specific price book
- OpportunityLineItem: Product on an opportunity

Setup Process:
1. Create Products (Product2)
2. Create Price Books (standard + custom)
3. Add Products to Price Books (PricebookEntry)
4. Add Products to Opportunities

SOQL Examples:
```soql
-- Get all products
SELECT Id, Name, ProductCode, IsActive FROM Product2 WHERE IsActive = true

-- Get price book entries
SELECT Id, Product2.Name, UnitPrice, Pricebook2.Name
FROM PricebookEntry
WHERE IsActive = true

-- Get opportunity line items with product details
SELECT Id, Name, Quantity, UnitPrice, TotalPrice, Product2.Name
FROM OpportunityLineItem
WHERE OpportunityId = '006xxxxxxxxxxxx'
```

Apex Example:
```apex
// Add product to opportunity
OpportunityLineItem oli = new OpportunityLineItem(
    OpportunityId = oppId,
    Product2Id = productId,
    PricebookEntryId = pbeId,
    Quantity = 10,
    UnitPrice = 100.00
);
insert oli;
```

Best Practice: Always use the Standard Price Book as a foundation, then create custom price books for different regions or customer segments.""",
            "source": "help_docs",
            "category": "sales",
            "subcategory": "products",
            "object_types": ["Opportunity", "Product2", "PricebookEntry", "OpportunityLineItem"],
            "features": ["Products", "Price Books", "Opportunity Line Items"],
            "is_code": True,
            "is_apex": True,
            "is_soql": True,
            "difficulty": "intermediate",
            "tags": ["products", "price book", "line items", "opportunity", "apex", "soql"],
            "keywords": ["product", "price book", "line item", "opportunity product"]
        },

        # === AUTOMATION - FLOWS ===
        {
            "title": "Record-Triggered Flows Best Practices",
            "content": """Record-Triggered Flows are the recommended way to automate actions when records are created, updated, or deleted.

Flow Types:
- Before Save: Runs before the record is saved (no DML operations)
- After Save: Runs after the record is saved (can perform DML)
- Before Delete: Runs before record deletion
- After Delete: Runs after record deletion

Best Practices:

1. **One Flow Per Object Per Trigger Type**
   Consolidate logic into a single flow per object/trigger combination to avoid conflicts.

2. **Use Before Save When Possible**
   - Faster (no extra DML)
   - Can update the triggering record directly
   - Use for field updates, validations

3. **Use After Save For Related Records**
   - Required when creating/updating other records
   - Required for sending emails
   - Required for calling external services

4. **Entry Conditions**
   Always use entry conditions to filter records:
   - Check if field changed: {!$Record.Field__c} != {!$Record__Prior.Field__c}
   - Check for specific values

5. **Bulkification**
   Flows handle bulkification automatically, but avoid:
   - Get/Update inside loops
   - Hard-coded limits

Example Flow: Update Account Rating When Opportunity Closes
1. Object: Opportunity
2. Trigger: After Save
3. Entry Condition: StageName = 'Closed Won' AND ISNEW()
4. Get Records: Account where Id = Opportunity.AccountId
5. Assignment: Account.Rating = 'Hot'
6. Update Records: Account

Common Errors:
- FLOW_LOOP: Recursive flow updates same record
- LIMIT_EXCEEDED: Too many SOQL/DML in transaction
- NULL_REFERENCE: Field path not checked for null""",
            "source": "trailhead",
            "category": "automation",
            "subcategory": "flows",
            "object_types": ["Opportunity", "Account"],
            "features": ["Flow", "Record-Triggered Flow", "Automation"],
            "is_flow": True,
            "difficulty": "intermediate",
            "tags": ["flow", "automation", "record-triggered", "best practices"],
            "keywords": ["flow", "trigger", "automation", "before save", "after save"]
        },
        {
            "title": "Screen Flows for User Input",
            "content": """Screen Flows provide interactive user interfaces within Salesforce.

Use Cases:
- Guided wizards
- Data entry forms
- Approval processes
- Quick actions

Key Components:

1. **Screen Elements**
   - Display Text: Rich text/merge fields
   - Input Fields: Text, number, date, picklist
   - Radio Buttons/Checkboxes
   - Lookup: Search for records
   - Data Table: Display/select records
   - File Upload: Attach files

2. **Variables**
   - Text, Number, Date, Boolean
   - Record Variables (single/collection)
   - Available for Input/Output

3. **Flow Actions**
   - Get Records: Query database
   - Create/Update/Delete Records
   - Send Email
   - Call Apex
   - Subflow: Call another flow

Building a Screen Flow:
1. Setup > Flows > New Flow > Screen Flow
2. Add Screen element
3. Configure input fields
4. Add logic (Decision, Loop, Assignment)
5. Add DML actions
6. Save and Activate
7. Add to Lightning Page, Quick Action, or Experience Site

Example: New Case Quick Entry
```
Screen 1: Case Details
- Subject (Text Input, Required)
- Description (Long Text Area)
- Priority (Picklist)
- Contact Lookup

Create Records: Case
- Subject = {!Subject}
- Description = {!Description}
- Priority = {!Priority}
- ContactId = {!Contact.Id}

Screen 2: Confirmation
- Display Text: "Case {!Case.CaseNumber} created successfully!"
```

Best Practice: Use input validation on screens to catch errors early. Keep flows simple - break complex processes into subflows.""",
            "source": "trailhead",
            "category": "automation",
            "subcategory": "flows",
            "object_types": ["Case"],
            "features": ["Screen Flow", "User Interface", "Quick Action"],
            "is_flow": True,
            "difficulty": "intermediate",
            "tags": ["screen flow", "wizard", "user interface", "input"],
            "keywords": ["screen flow", "wizard", "form", "input", "user interface"]
        },

        # === APEX DEVELOPMENT ===
        {
            "title": "Apex Triggers Best Practices",
            "content": """Apex Triggers execute before or after DML operations on Salesforce records.

Trigger Context Variables:
- Trigger.new: List of new records (insert/update)
- Trigger.old: List of old records (update/delete)
- Trigger.newMap: Map of ID to new records
- Trigger.oldMap: Map of ID to old records
- Trigger.isInsert, isUpdate, isDelete, isUndelete
- Trigger.isBefore, isAfter

Best Practices:

1. **One Trigger Per Object**
```apex
trigger AccountTrigger on Account (before insert, before update, after insert, after update, before delete, after delete) {
    AccountTriggerHandler handler = new AccountTriggerHandler();

    if (Trigger.isBefore) {
        if (Trigger.isInsert) handler.beforeInsert(Trigger.new);
        if (Trigger.isUpdate) handler.beforeUpdate(Trigger.new, Trigger.oldMap);
        if (Trigger.isDelete) handler.beforeDelete(Trigger.old);
    } else {
        if (Trigger.isInsert) handler.afterInsert(Trigger.new);
        if (Trigger.isUpdate) handler.afterUpdate(Trigger.new, Trigger.oldMap);
        if (Trigger.isDelete) handler.afterDelete(Trigger.old);
    }
}
```

2. **Bulkification**
```apex
// BAD - Query in loop
for (Account acc : Trigger.new) {
    Contact c = [SELECT Id FROM Contact WHERE AccountId = :acc.Id LIMIT 1];
}

// GOOD - Bulkified
Set<Id> accountIds = new Set<Id>();
for (Account acc : Trigger.new) {
    accountIds.add(acc.Id);
}
Map<Id, Contact> contactsByAccount = new Map<Id, Contact>();
for (Contact c : [SELECT Id, AccountId FROM Contact WHERE AccountId IN :accountIds]) {
    contactsByAccount.put(c.AccountId, c);
}
```

3. **Recursion Prevention**
```apex
public class TriggerHandler {
    private static Boolean isExecuting = false;

    public static Boolean isFirstRun() {
        if (isExecuting) return false;
        isExecuting = true;
        return true;
    }

    public static void reset() {
        isExecuting = false;
    }
}
```

4. **Governor Limit Awareness**
- 100 SOQL queries per transaction
- 150 DML statements
- 6MB heap size
- 10 seconds CPU time""",
            "source": "apex_guide",
            "category": "development",
            "subcategory": "triggers",
            "object_types": ["Account", "Contact"],
            "features": ["Apex", "Triggers", "Handler Pattern"],
            "is_code": True,
            "is_apex": True,
            "difficulty": "advanced",
            "tags": ["apex", "trigger", "best practices", "bulkification", "governor limits"],
            "keywords": ["trigger", "apex", "before", "after", "bulkify", "handler"]
        },
        {
            "title": "SOQL Query Optimization",
            "content": """Optimize SOQL queries for better performance and to avoid governor limits.

Key Optimization Techniques:

1. **Use Selective Filters**
```soql
-- BAD: Full table scan
SELECT Id, Name FROM Account

-- GOOD: Indexed field filter
SELECT Id, Name FROM Account WHERE CreatedDate = LAST_N_DAYS:7

-- Indexed fields: Id, Name, OwnerId, CreatedDate, LastModifiedDate, SystemModstamp
-- Custom indexed fields (request from Salesforce)
```

2. **Limit Returned Fields**
```soql
-- BAD: Select all fields
SELECT FIELDS(ALL) FROM Account LIMIT 200

-- GOOD: Only needed fields
SELECT Id, Name, Industry FROM Account LIMIT 200
```

3. **Use Relationship Queries**
```soql
-- Parent-to-child (subquery)
SELECT Id, Name, (SELECT Id, FirstName, LastName FROM Contacts)
FROM Account
WHERE Industry = 'Technology'

-- Child-to-parent (dot notation)
SELECT Id, FirstName, LastName, Account.Name, Account.Industry
FROM Contact
WHERE Account.Industry = 'Technology'
```

4. **Avoid Negative Operators**
```soql
-- BAD: Not selective
SELECT Id FROM Account WHERE Industry != 'Technology'

-- GOOD: Use positive filter
SELECT Id FROM Account WHERE Industry = 'Healthcare'
```

5. **Use LIMIT and OFFSET**
```soql
-- Pagination
SELECT Id, Name FROM Account ORDER BY Name LIMIT 100 OFFSET 200
```

6. **Aggregate Queries**
```soql
-- Count records
SELECT COUNT() FROM Opportunity WHERE StageName = 'Closed Won'

-- Group by with aggregates
SELECT StageName, COUNT(Id), SUM(Amount)
FROM Opportunity
GROUP BY StageName
HAVING COUNT(Id) > 10
```

Query Plan Tool:
1. Developer Console > Query Editor
2. Click "Query Plan"
3. Look for:
   - Leading operation type (Index vs TableScan)
   - Cardinality (number of records)
   - Cost (lower is better)""",
            "source": "apex_guide",
            "category": "development",
            "subcategory": "soql",
            "object_types": ["Account", "Contact", "Opportunity"],
            "features": ["SOQL", "Query Optimization", "Performance"],
            "is_code": True,
            "is_soql": True,
            "difficulty": "advanced",
            "tags": ["soql", "query", "optimization", "performance", "indexes"],
            "keywords": ["soql", "query", "optimize", "index", "selective", "performance"]
        },

        # === REPORTS AND DASHBOARDS ===
        {
            "title": "Creating Reports in Salesforce",
            "content": """Salesforce Reports provide insights into your data with filtering, grouping, and visualization.

Report Types:
1. **Tabular**: Simple list, no grouping
2. **Summary**: Group by rows
3. **Matrix**: Group by rows and columns
4. **Joined**: Combine multiple report types

Creating a Report:
1. App Launcher > Reports
2. New Report
3. Select Report Type (determines available objects/fields)
4. Add columns (fields to display)
5. Add filters
6. Add groupings
7. Run and Save

Common Report Customizations:

Filters:
- Standard filters: Date range, owner, record type
- Field filters: Field operator value
- Filter logic: AND/OR combinations
- Cross-filter: "Accounts with/without Opportunities"

Groupings:
- Row groupings: Up to 3 levels
- Column groupings: For matrix reports
- Group date fields by: Day, Week, Month, Quarter, Year

Formulas:
- Summary formulas: Calculate across groups
- Row-level formulas: Calculate per row

Example: Opportunity Pipeline Report
1. Report Type: Opportunities
2. Columns: Opportunity Name, Account Name, Amount, Close Date, Stage
3. Filter: Close Date = THIS FISCAL QUARTER
4. Grouping: Stage Name
5. Summary: SUM(Amount) per stage

Report Scheduling:
1. Open saved report
2. Subscribe
3. Set frequency (daily, weekly, monthly)
4. Choose conditions (always or when conditions met)
5. Select recipients

Exporting Reports:
- Export > Formatted Report (Excel)
- Export > Details Only (CSV)
- Printable View (PDF)

Best Practice: Create report folders by team/function. Use naming conventions like "Sales - Pipeline Reports".""",
            "source": "help_docs",
            "category": "reporting",
            "subcategory": "reports",
            "object_types": ["Opportunity", "Account"],
            "features": ["Reports", "Analytics"],
            "difficulty": "beginner",
            "tags": ["reports", "analytics", "dashboard", "filters", "grouping"],
            "keywords": ["report", "analytics", "filter", "group", "summary", "matrix"]
        },

        # === SECURITY ===
        {
            "title": "Salesforce Security Model Overview",
            "content": """Salesforce security operates at multiple levels to control data access.

Security Layers (from broad to specific):

1. **Organization-Wide Defaults (OWD)**
   - Private: Only owner and above can access
   - Public Read Only: All can view, only owner can edit
   - Public Read/Write: All can view and edit
   - Controlled by Parent: Follows parent object's sharing

   Setting OWD: Setup > Sharing Settings

2. **Role Hierarchy**
   - Users inherit access from roles below them
   - CEO sees all, reps see only their own

   Setting Roles: Setup > Roles

3. **Sharing Rules**
   - Extend access beyond OWD
   - Based on ownership or criteria

   Types:
   - Owner-based: Share records owned by users in Group A with Group B
   - Criteria-based: Share records where Region = 'West' with West Team

4. **Manual Sharing**
   - Users share individual records
   - Granted Read Only or Read/Write access

5. **Teams**
   - Account Teams, Opportunity Teams, Case Teams
   - Named users with specific access levels

6. **Profiles & Permission Sets**
   - Object permissions: CRUD (Create, Read, Update, Delete)
   - Field-level security: Visible, Read-Only, Hidden

   Best Practice: Minimal profile + Permission Sets for additional access

7. **Permission Set Groups**
   - Bundle related permission sets
   - Muting permission sets to remove specific permissions

Apex Sharing:
```apex
// Share record programmatically
AccountShare share = new AccountShare(
    AccountId = accountId,
    UserOrGroupId = userId,
    AccountAccessLevel = 'Edit',
    OpportunityAccessLevel = 'Read',
    RowCause = Schema.AccountShare.RowCause.Manual
);
insert share;
```

Checking Access in Apex:
```apex
// Check object access
Schema.DescribeSObjectResult describeResult = Account.sObjectType.getDescribe();
Boolean canCreate = describeResult.isCreateable();
Boolean canRead = describeResult.isAccessible();

// Check field access
Schema.DescribeFieldResult fieldResult = Account.Industry.getDescribe();
Boolean fieldVisible = fieldResult.isAccessible();
```

Security Best Practices:
- Start with Private OWD, open up with sharing rules
- Use Permission Sets over Profiles for flexibility
- Regular access reviews with Permission Set assignments
- Use "with sharing" keyword in Apex by default""",
            "source": "admin_guide",
            "category": "security",
            "subcategory": "sharing",
            "object_types": ["Account", "Opportunity", "Case"],
            "features": ["Sharing", "Security", "Profiles", "Permission Sets"],
            "is_code": True,
            "is_apex": True,
            "difficulty": "advanced",
            "tags": ["security", "sharing", "owd", "roles", "profiles", "permission sets"],
            "keywords": ["security", "sharing", "access", "permission", "profile", "role"]
        },

        # === INTEGRATION ===
        {
            "title": "REST API Basics for Salesforce",
            "content": """Salesforce REST API allows external systems to interact with Salesforce data.

Authentication:
1. **Username-Password Flow** (server-to-server)
2. **Web Server Flow** (user authorization)
3. **JWT Bearer Flow** (server-to-server, no secret)

OAuth 2.0 Token Request:
```
POST /services/oauth2/token
Content-Type: application/x-www-form-urlencoded

grant_type=password
&client_id=YOUR_CONSUMER_KEY
&client_secret=YOUR_CONSUMER_SECRET
&username=user@example.com
&password=passwordSECURITY_TOKEN
```

Common Endpoints:

Query Records:
```
GET /services/data/v59.0/query?q=SELECT+Id,Name+FROM+Account+LIMIT+10
Authorization: Bearer ACCESS_TOKEN
```

Get Record by ID:
```
GET /services/data/v59.0/sobjects/Account/001xxxxxxxxxxxx
Authorization: Bearer ACCESS_TOKEN
```

Create Record:
```
POST /services/data/v59.0/sobjects/Account
Authorization: Bearer ACCESS_TOKEN
Content-Type: application/json

{
    "Name": "New Account",
    "Industry": "Technology"
}
```

Update Record:
```
PATCH /services/data/v59.0/sobjects/Account/001xxxxxxxxxxxx
Authorization: Bearer ACCESS_TOKEN
Content-Type: application/json

{
    "Industry": "Healthcare"
}
```

Delete Record:
```
DELETE /services/data/v59.0/sobjects/Account/001xxxxxxxxxxxx
Authorization: Bearer ACCESS_TOKEN
```

Composite API (multiple operations):
```
POST /services/data/v59.0/composite
Authorization: Bearer ACCESS_TOKEN
Content-Type: application/json

{
    "compositeRequest": [
        {
            "method": "POST",
            "url": "/services/data/v59.0/sobjects/Account",
            "referenceId": "newAccount",
            "body": {"Name": "New Account"}
        },
        {
            "method": "POST",
            "url": "/services/data/v59.0/sobjects/Contact",
            "referenceId": "newContact",
            "body": {
                "FirstName": "John",
                "LastName": "Doe",
                "AccountId": "@{newAccount.id}"
            }
        }
    ]
}
```

Rate Limits:
- API requests: Based on edition and licenses
- Concurrent API limit: 25 requests
- Check with: GET /services/data/v59.0/limits""",
            "source": "apex_guide",
            "category": "integration",
            "subcategory": "rest_api",
            "features": ["REST API", "Integration", "OAuth"],
            "is_code": True,
            "difficulty": "advanced",
            "tags": ["rest api", "integration", "oauth", "http", "endpoints"],
            "keywords": ["rest", "api", "integration", "oauth", "http", "endpoint"]
        },

        # === LIGHTNING COMPONENTS ===
        {
            "title": "Lightning Web Components Basics",
            "content": """Lightning Web Components (LWC) is Salesforce's modern UI framework based on web standards.

File Structure:
```
myComponent/
├── myComponent.html      # Template
├── myComponent.js        # JavaScript controller
├── myComponent.css       # Styles (optional)
├── myComponent.js-meta.xml  # Metadata configuration
```

Basic Component:

myComponent.html:
```html
<template>
    <lightning-card title="My Component">
        <p class="slds-p-horizontal_small">
            Hello, {greeting}!
        </p>
        <lightning-button
            label="Click Me"
            onclick={handleClick}>
        </lightning-button>
    </lightning-card>
</template>
```

myComponent.js:
```javascript
import { LightningElement, api, track, wire } from 'lwc';
import getAccounts from '@salesforce/apex/AccountController.getAccounts';

export default class MyComponent extends LightningElement {
    @api recordId;  // Public property (from parent or record page)
    @track greeting = 'World';  // Reactive property
    accounts = [];
    error;

    // Wire service - automatic data fetching
    @wire(getAccounts)
    wiredAccounts({ error, data }) {
        if (data) {
            this.accounts = data;
            this.error = undefined;
        } else if (error) {
            this.error = error;
            this.accounts = [];
        }
    }

    handleClick() {
        this.greeting = 'Salesforce';
    }

    // Lifecycle hooks
    connectedCallback() {
        console.log('Component connected to DOM');
    }

    renderedCallback() {
        console.log('Component rendered');
    }
}
```

myComponent.js-meta.xml:
```xml
<?xml version="1.0" encoding="UTF-8"?>
<LightningComponentBundle xmlns="http://soap.sforce.com/2006/04/metadata">
    <apiVersion>59.0</apiVersion>
    <isExposed>true</isExposed>
    <targets>
        <target>lightning__RecordPage</target>
        <target>lightning__AppPage</target>
        <target>lightning__HomePage</target>
    </targets>
    <targetConfigs>
        <targetConfig targets="lightning__RecordPage">
            <objects>
                <object>Account</object>
            </objects>
        </targetConfig>
    </targetConfigs>
</LightningComponentBundle>
```

Apex Controller:
```apex
public with sharing class AccountController {
    @AuraEnabled(cacheable=true)
    public static List<Account> getAccounts() {
        return [SELECT Id, Name, Industry FROM Account LIMIT 10];
    }
}
```

Key Concepts:
- @api: Public properties/methods
- @track: Reactive properties (deprecated in most cases, reactivity is automatic)
- @wire: Declarative data fetching
- Lifecycle hooks: connectedCallback, renderedCallback, disconnectedCallback""",
            "source": "apex_guide",
            "category": "development",
            "subcategory": "lwc",
            "features": ["Lightning Web Components", "LWC", "UI"],
            "is_code": True,
            "is_apex": True,
            "difficulty": "advanced",
            "tags": ["lwc", "lightning", "component", "javascript", "apex"],
            "keywords": ["lwc", "lightning", "component", "wire", "apex", "javascript"]
        }
    ]

    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()

    print(f"\nInserting {len(sample_docs)} Salesforce documents...")

    for doc in sample_docs:
        # Generate embedding
        embedding = model.encode(doc["content"], normalize_embeddings=True)

        cur.execute("""
            INSERT INTO salesforce_documents (
                title, content, embedding, source, category, subcategory,
                object_types, features, is_code, is_apex, is_soql, is_flow,
                difficulty, tags, keywords, salesforce_version
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id
        """, (
            doc["title"],
            doc["content"],
            embedding.tolist(),
            doc.get("source", "help_docs"),
            doc.get("category", "general"),
            doc.get("subcategory"),
            doc.get("object_types", []),
            doc.get("features", []),
            doc.get("is_code", False),
            doc.get("is_apex", False),
            doc.get("is_soql", False),
            doc.get("is_flow", False),
            doc.get("difficulty", "intermediate"),
            doc.get("tags", []),
            doc.get("keywords", []),
            "Winter 25"
        ))

        doc_id = cur.fetchone()[0]
        print(f"  ✓ Inserted: {doc['title'][:50]}... (ID: {doc_id})")

    conn.commit()
    cur.close()
    conn.close()

    print(f"\n✅ Inserted {len(sample_docs)} Salesforce documents with embeddings!")


def verify_salesforce_setup():
    """Verify the Salesforce database setup"""
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()

    print("\n=== Salesforce Database Verification ===")

    # Check document count
    cur.execute("SELECT COUNT(*) FROM salesforce_documents")
    count = cur.fetchone()[0]
    print(f"Documents: {count}")

    # Check categories
    cur.execute("""
        SELECT category, COUNT(*)
        FROM salesforce_documents
        GROUP BY category
        ORDER BY COUNT(*) DESC
    """)
    categories = cur.fetchall()
    print(f"Categories: {dict(categories)}")

    # Check vector index
    cur.execute("""
        SELECT indexname FROM pg_indexes
        WHERE tablename = 'salesforce_documents' AND indexdef LIKE '%hnsw%'
    """)
    hnsw_index = cur.fetchone()
    print(f"HNSW Index: {'✓' if hnsw_index else '✗'}")

    # Check text search index
    cur.execute("""
        SELECT indexname FROM pg_indexes
        WHERE tablename = 'salesforce_documents' AND indexdef LIKE '%gin%'
    """)
    gin_index = cur.fetchone()
    print(f"GIN Index: {'✓' if gin_index else '✗'}")

    # Test vector search
    cur.execute("""
        SELECT id, title, 1 - (embedding <=> embedding) as self_sim
        FROM salesforce_documents
        WHERE embedding IS NOT NULL
        LIMIT 1
    """)
    result = cur.fetchone()
    if result:
        print(f"Vector Search Test: ✓ (self-similarity = {result[2]:.4f})")

    # Test text search
    cur.execute("""
        SELECT id, title, ts_rank(tsv, plainto_tsquery('english', 'custom field account')) as rank
        FROM salesforce_documents
        WHERE tsv @@ plainto_tsquery('english', 'custom field account')
        ORDER BY rank DESC
        LIMIT 1
    """)
    result = cur.fetchone()
    if result:
        print(f"Text Search Test: ✓ (top result: {result[1][:40]}...)")

    cur.close()
    conn.close()

    print("\n✅ Salesforce database verification complete!")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--verify":
        verify_salesforce_setup()
    elif len(sys.argv) > 1 and sys.argv[1] == "--sample":
        insert_sample_salesforce_documents()
    elif len(sys.argv) > 1 and sys.argv[1] == "--schema-only":
        create_salesforce_schema()
    else:
        create_salesforce_schema()
        insert_sample_salesforce_documents()
        verify_salesforce_setup()
