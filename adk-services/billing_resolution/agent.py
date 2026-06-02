import sqlite3
from datetime import datetime
from typing import Dict, Any
import os 

from dotenv import load_dotenv

from google.adk.agents import Agent
from google.adk.a2a.utils.agent_to_a2a import to_a2a

import uvicorn

# ---------------------------------------------------
# Load Environment Variables
# ---------------------------------------------------
load_dotenv()

# ---------------------------------------------------
# Database Path
# ---------------------------------------------------
DATABASE_PATH = r"c:\Users\Admin\capstone_project\data\telecom_ops.db"

# ---------------------------------------------------
# Auto Approval Policy
# ---------------------------------------------------
AUTO_APPROVAL_LIMIT = 50.0

# ---------------------------------------------------
# Tool 1
# Lookup Billing Account
# ---------------------------------------------------
async def lookup_billing_account(
    customer_id: str,
) -> Dict[str, Any]:

    conn = sqlite3.connect(DATABASE_PATH)

    conn.row_factory = sqlite3.Row

    cursor = conn.cursor()

    # -------------------------------------------
    # Billing account
    # -------------------------------------------
    cursor.execute(
        """
        SELECT *
        FROM billing_accounts
        WHERE customer_id = ?
        """,
        (customer_id,),
    )

    account = cursor.fetchone()

    if not account:

        conn.close()

        return {
            "error": f"Customer {customer_id} not found."
        }

    # -------------------------------------------
    # Recent charges
    # -------------------------------------------
    cursor.execute(
        """
        SELECT *
        FROM billing_charges
        WHERE customer_id = ?
        ORDER BY charge_date DESC
        LIMIT 5
        """,
        (customer_id,),
    )

    charges = cursor.fetchall()

    conn.close()

    return {
        "customer_id": customer_id,
        "billing_account": dict(account),
        "recent_charges": [
            dict(charge)
            for charge in charges
        ],
    }


# ---------------------------------------------------
# Tool 2
# Check Duplicate Charges
# ---------------------------------------------------
async def check_duplicate_charges(
    customer_id: str,
) -> Dict[str, Any]:

    conn = sqlite3.connect(DATABASE_PATH)

    conn.row_factory = sqlite3.Row

    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT *
        FROM billing_charges
        WHERE customer_id = ?
        AND is_duplicate_flag = 1
        """,
        (customer_id,),
    )

    duplicates = cursor.fetchall()

    conn.close()

    return {
        "customer_id": customer_id,
        "duplicate_charges_found": len(duplicates),
        "duplicate_charges": [
            dict(charge)
            for charge in duplicates
        ],
    }


# ---------------------------------------------------
# Tool 3
# Apply Billing Credit
# ---------------------------------------------------
async def apply_billing_credit(
    customer_id: str,
    amount: float,
    reason: str,
) -> Dict[str, Any]:

    conn = sqlite3.connect(DATABASE_PATH)

    conn.row_factory = sqlite3.Row

    cursor = conn.cursor()

    # -------------------------------------------
    # Determine approval status
    # -------------------------------------------
    if amount > AUTO_APPROVAL_LIMIT:

        status = "PENDING_APPROVAL"

    else:

        status = "APPLIED"

    reference_number = (
        f"CR-{datetime.now().strftime('%Y%m%d%H%M%S')}"
    )

    applied_at = datetime.now().isoformat()

    # -------------------------------------------
    # Insert billing credit
    # -------------------------------------------
    cursor.execute(
        """
        INSERT INTO billing_credits (
            customer_id,
            amount,
            reason,
            status,
            reference_number,
            applied_at
        )
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            customer_id,
            amount,
            reason,
            status,
            reference_number,
            applied_at,
        ),
    )

    # -------------------------------------------
    # Update billing disputes
    # -------------------------------------------
    cursor.execute(
        """
        UPDATE billing_disputes
        SET status = 'RESOLVED',
            resolved_at = ?
        WHERE customer_id = ?
          AND status = 'OPEN'
        """,
        (
            applied_at,
            customer_id,
        ),
    )

    # -------------------------------------------
    # Update balance ONLY if auto-approved
    # -------------------------------------------
    if status == "APPLIED":

        cursor.execute(
            """
            UPDATE billing_accounts
            SET current_balance =
                current_balance - ?
            WHERE customer_id = ?
            """,
            (
                amount,
                customer_id,
            ),
        )

    conn.commit()

    conn.close()

    return {
        "customer_id": customer_id,
        "credit_amount": amount,
        "reason": reason,
        "status": status,
        "reference_number": reference_number,
    }


# ---------------------------------------------------
# Billing Resolution Agent
# ---------------------------------------------------
billing_resolution_agent = Agent(
    name="billing_resolution_agent",

    model="gpt-4o-mini",

    description=(
        "Telecom billing dispute resolution agent."
    ),

    instruction=(
        "You are a telecom billing resolution agent. "
        "Help investigate billing disputes, "
        "duplicate charges, refunds, credits, "
        "account balances, and billing issues. "
        "Use the available SQL-backed tools "
        "to retrieve and update billing data."
    ),

    tools=[
        lookup_billing_account,
        check_duplicate_charges,
        apply_billing_credit,
    ],
)


# ---------------------------------------------------
# Convert to A2A App
# ---------------------------------------------------
app = to_a2a(
    billing_resolution_agent,
    port=8002
)

# ---------------------------------------------------
# Run Server
# ---------------------------------------------------
if __name__ == "__main__":

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8002,
    )