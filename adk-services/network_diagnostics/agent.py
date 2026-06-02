
import os
import sqlite3

from typing import Dict, Any

from dotenv import load_dotenv

from google.adk.agents import Agent
from google.adk.a2a.utils.agent_to_a2a import to_a2a


import uvicorn

# ---------------------------------------------------
# Load Environment Variables
# ---------------------------------------------------
load_dotenv()

# ---------------------------------------------------
# OpenAI API Key
# ---------------------------------------------------
os.environ["OPENAI_API_KEY"] = os.getenv(
    "OPENAI_API_KEY"
)

# ---------------------------------------------------
# Absolute Database Path
# ---------------------------------------------------
DATABASE_PATH = r"c:\Users\Admin\capstone_project\data\telecom_ops.db"

# ---------------------------------------------------
# Tool 1
# Check Tower Status
# ---------------------------------------------------
async def check_tower_status(
    tower_id: str,
) -> Dict[str, Any]:

    conn = sqlite3.connect(DATABASE_PATH)

    conn.row_factory = sqlite3.Row

    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT *
        FROM network_towers
        WHERE tower_id = ?
        """,
        (tower_id,),
    )

    tower = cursor.fetchone()

    if not tower:

        conn.close()

        return {
            "error": f"Tower {tower_id} not found."
        }

    cursor.execute(
        """
        SELECT *
        FROM tower_performance
        WHERE tower_id = ?
        ORDER BY recorded_at DESC
        LIMIT 1
        """,
        (tower_id,),
    )

    performance = cursor.fetchone()

    cursor.execute(
        """
        SELECT *
        FROM open_incidents
        WHERE tower_id = ?
        AND status != 'RESOLVED'
        """,
        (tower_id,),
    )

    incidents = cursor.fetchall()

    conn.close()

    return {
        "tower_info": dict(tower),

        "latest_performance": (
            dict(performance)
            if performance
            else None
        ),

        "open_incidents": [
            dict(i)
            for i in incidents
        ],
    }

# ---------------------------------------------------
# Tool 2
# Connectivity Diagnostics
# ---------------------------------------------------
async def run_connectivity_diagnostics(
    tower_id: str,
    symptom: str,
) -> Dict[str, Any]:

    conn = sqlite3.connect(DATABASE_PATH)

    conn.row_factory = sqlite3.Row

    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT *
        FROM network_towers
        WHERE tower_id = ?
        """,
        (tower_id,),
    )

    tower = cursor.fetchone()

    if not tower:

        conn.close()

        return {
            "error": f"Tower {tower_id} not found."
        }

    cursor.execute(
        """
        SELECT *
        FROM tower_performance
        WHERE tower_id = ?
        ORDER BY recorded_at DESC
        LIMIT 1
        """,
        (tower_id,),
    )

    performance = cursor.fetchone()

    recommendations = []

    if performance:

        latency = performance[
            "avg_latency_ms"
        ]

        packet_loss = performance[
            "packet_loss_pct"
        ]

        throughput = performance[
            "throughput_mbps"
        ]

        signal_strength = performance[
            "signal_strength_dbm"
        ]

        if latency and latency > 100:

            recommendations.append(
                "High latency detected."
            )

        if packet_loss and packet_loss > 2:

            recommendations.append(
                "Packet loss elevated."
            )

        if throughput and throughput < 50:

            recommendations.append(
                "Low throughput detected."
            )

        if (
            signal_strength
            and signal_strength < -90
        ):

            recommendations.append(
                "Weak signal strength detected."
            )

    if tower["status"] == "OUTAGE":

        recommendations.append(
            "Tower currently in outage."
        )

    if tower["status"] == "DEGRADED":

        recommendations.append(
            "Tower performance degraded."
        )

    conn.close()

    return {
        "tower_id": tower_id,
        "symptom": symptom,
        "tower_status": tower["status"],
        "recommendations": recommendations,
    }

# ---------------------------------------------------
# Tool 3
# Regional Summary
# ---------------------------------------------------
async def get_regional_network_summary(
    region: str,
) -> Dict[str, Any]:

    conn = sqlite3.connect(DATABASE_PATH)

    conn.row_factory = sqlite3.Row

    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT COUNT(*) AS total_towers
        FROM network_towers
        WHERE region = ?
        """,
        (region,),
    )

    total_towers = cursor.fetchone()[
        "total_towers"
    ]

    cursor.execute(
        """
        SELECT COUNT(*) AS operational
        FROM network_towers
        WHERE region = ?
        AND status = 'OPERATIONAL'
        """,
        (region,),
    )

    operational = cursor.fetchone()[
        "operational"
    ]

    cursor.execute(
        """
        SELECT COUNT(*) AS impacted
        FROM network_towers
        WHERE region = ?
        AND status IN (
            'DEGRADED',
            'OUTAGE'
        )
        """,
        (region,),
    )

    impacted = cursor.fetchone()[
        "impacted"
    ]

    conn.close()

    return {
        "region": region,
        "total_towers": total_towers,
        "operational_towers": operational,
        "impacted_towers": impacted,
    }

# ---------------------------------------------------
# ADK Agent
# ---------------------------------------------------
network_diagnostics_agent = Agent(

    name="network_diagnostics_agent",

    model="gpt-4o-mini",

    description=(
        "Telecom network diagnostics agent."
    ),

    instruction=(
        "You are a telecom NOC diagnostics "
        "assistant responsible for "
        "tower diagnostics, outages, "
        "and regional health analysis."
    ),

    tools=[
        check_tower_status,
        run_connectivity_diagnostics,
        get_regional_network_summary,
    ],
)

# ---------------------------------------------------
# A2A App
# ---------------------------------------------------
app = to_a2a(
    network_diagnostics_agent,
    port=8001
)

# ---------------------------------------------------
# Run Server
# ---------------------------------------------------
if __name__ == "__main__":

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8001,
    )
