import json
import time
from pathlib import Path

from utils.plot_trace import plot_trace
from utils.runners import run_session

from agents.CSE3210.agent68.utils.plot_pareto import plot_pareto, compute_pareto_frontier

RESULTS_DIR = Path("results", "Agent2")

# create results directory if it does not exist
if not RESULTS_DIR.exists():
    RESULTS_DIR.mkdir(parents=True)

domain = "domain12"
# Settings to run a negotiation session:
#   You need to specify the classpath of 2 agents to start a negotiation. Parameters for the agent can be added as a dict (see example)
#   You need to specify the preference profiles for both agents. The first profile will be assigned to the first agent.
#   You need to specify a time deadline (is milliseconds (ms)) we are allowed to negotiate before we end without agreement
settings = {
    "agents": [
        # {
        #     "class": "agents.Group46_Negotiation.agent_46.OurAgent",
        #     "parameters": {"storage_dir": "agent_storage/OurAgent2"},
        # },
        # {
        #     "class": "agents.hardliner_agent.hardliner_agent.HardlinerAgent",
        #     "parameters": {"storage_dir": "agent_storage/HardlinerAgent"},
        # },
        # {
        #     "class": "agents.time_dependent_agent.time_dependent_agent.TimeDependentAgent",
        #     "parameters": {"storage_dir": "agent_storage/TimeDependentAgent"},
        # },
        # {
        #     "class": "agents.linear_agent.linear_agent.LinearAgent",
        #     "parameters": {"storage_dir": "agent_storage/LinearAgent"},
        # },
        # {
        #     "class": "agents.random_agent.random_agent.RandomAgent",
        #     "parameters": {"storage_dir": "agent_storage/RandomAgent"},
        # },
        {
            "class": "agents.CSE3210.agent2.agent2.Agent2",
            "parameters": {"storage_dir": "agent_storage/Agent2"},
        },
        # {
        #     "class": "agents.boulware_agent.boulware_agent.BoulwareAgent",
        #     "parameters": {"storage_dir": "agent_storage/BoulwareAgent"},
        # },
        # {
        #     "class": "agents.conceder_agent.conceder_agent.ConcederAgent",
        #     "parameters": {"storage_dir": "agent_storage/ConcederAgent"},
        # },
        # {
        #     "class": "agents.Group46_Negotiation.agent_46.OurAgent",
        #     "parameters": {"storage_dir": "agent_storage/OurAgent"},
        # },
        # {
        #     "class": "agents.ANL2022.dreamteam109_agent.dreamteam109_agent.DreamTeam109Agent",
        #     "parameters": {"storage_dir": "agent_storage/DreamTeam109Agent"},
        # },
        {
            "class": "agents.Group46_Negotiation.agent_46.OurAgent",
            "parameters": {"storage_dir": "agent_storage/OurAgent"},
        },

    ],
    "profiles": ["domains/domain00/profileA.json", "domains/domain00/profileB.json"],
    "deadline_time_ms": 10000,
}

# run a session and obtain results in dictionaries
session_results_trace, session_results_summary = run_session(settings)

plot_pareto(session_results_trace, compute_pareto_frontier([f"domains/{domain}/profileA.json", f"domains/{domain}/profileB.json"]),"pareto.html")

print(session_results_summary)
# plot trace to html file
if not session_results_trace["error"]:
    plot_trace(session_results_trace, RESULTS_DIR.joinpath("trace_plot.html"))

# write results to file
# with open(RESULTS_DIR.joinpath("session_results_trace.json"), "w", encoding="utf-8") as f:
#     f.write(json.dumps(session_results_trace, indent=2))
# with open(RESULTS_DIR.joinpath("session_results_summary.json"), "w", encoding="utf-8") as f:
#     f.write(json.dumps(session_results_summary, indent=2))

my_file = Path(RESULTS_DIR.joinpath("session_results_summary.json"))
if not my_file.is_file():
    session_results_summary["counter"] = 1
    if (session_results_summary["result"] == "agreement"):
        session_results_summary["agreement_counter"] = 1
    else:
        session_results_summary["agreement_counter"] = 0
    with open(RESULTS_DIR.joinpath("session_results_summary.json"), "w", encoding="utf-8") as f:
        f.write(json.dumps(session_results_summary, indent=2))
else:
    f = open(RESULTS_DIR.joinpath("session_results_summary.json"))
    content = json.load(f)
    counter = content["counter"]
    updated_counter = counter + 1
    session_results_summary["counter"] = updated_counter
    f.close()
    if(session_results_summary["result"] == "agreement"):
        session_results_summary["agreement_counter"] = content["agreement_counter"] + 1
    else:
        session_results_summary["agreement_counter"] = content["agreement_counter"]
        session_results_summary["utility_2"] = content["utility_2"]
        session_results_summary["nash_product"] = content["nash_product"]
        session_results_summary["social_welfare"] = content["social_welfare"]

    with open(RESULTS_DIR.joinpath("session_results_summary.json"),"w", encoding="utf-8") as f:
        f.write(json.dumps(session_results_summary, indent=2))