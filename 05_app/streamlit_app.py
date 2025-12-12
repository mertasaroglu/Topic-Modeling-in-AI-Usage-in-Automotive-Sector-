# AUTOMOTIVE TECH INNOVATION INTELLIGENCE SUITE - STREAMLIT APP

import streamlit as st
import sys
import os
import importlib.util
import re
import matplotlib.pyplot as plt

# Technology definitions dictionary
AUTO_TOP_SEEDS_1 = {
    "Sensor_Fusion": "multi sensor fusion architecture combining lidar radar and camera into unified perception outputs",
    "Occupancy_Grid": "spatial occupancy grid mapping for free space and obstacle representation around the vehicle",
    "SLAM": "simultaneous localization and mapping using onboard sensors for ego pose and map building in dynamic traffic",
    "Trajectory_Prediction": "future motion and path prediction of vehicles and vulnerable road users in traffic scenes",
    "Environment_Modeling": "semantic scene and object relationship modeling for a structured driving environment representation"
}
AUTO_TOP_SEEDS_2 = {
    "4G": "4g lte cellular communication for connected vehicles",
    "5G": "5g cellular communication for ultra low latency vehicle connectivity",
    "Wireless_Communication": "wireless communication protocols for vehicle data transmission",
    "V2V": "vehicle to vehicle direct communication",
    "V2I": "vehicle to infrastructure communication",
    "V2X": "v2x communication framework including v2v and v2i",
    "Edge_Computing": "edge computing for real time vehicle data processing",
    "Fog_Computing": "fog computing layer between edge and cloud",
    "Cloud_Computing": "cloud computing backend for vehicle data storage and processing"
}
AUTO_TOP_SEEDS_3 = {
    "Renewable_Energy": "renewable energy systems",
    "Solar_Cell": "photovoltaic solar cells (solar cell, perovskite solar, sensitized solar, organic photovoltaics, quantum dots, silicon solar)",
    "Electrochemical_Energy": "electrochemical energy systems (electrochemical energy, hydrogen evolution)",
    "Nano_Energy": "nanoscale energy systems (nano energy, quantum dots)",
    "Natural_Gas_Energy": "natural gas based energy systems (natural gas)",
}
AUTO_TOP_SEEDS_4 = {
    "Battery_Management_System": "battery management and control (battery management bms, battery management system, state charge soc, state health soh, battery state health)",
    "High_Energy_Batteries": "lithium ion battery technology (li ion batteries, ion batteries electric, performance lithium ion, high energy density)",
    "Battery_Thermal_Management": "battery thermal and cooling systems (battery thermal management, thermal management systems)",
    "Battery_Diagnostics_EIS": "electrochemical battery diagnostics (electrochemical impedance spectroscopy)",
    "Battery_Performance_Prediction": "lithium ion aging and performance prediction (prediction lithium ion, RUL)"
}
AUTO_TOP_SEEDS_5 = {
    "Smart_Grid": "smart grid control and monitoring (smart grid technologies, power grid)",
    "Distributed_Energy_Resources": "distributed energy generation and control (distributed energy resources)",
    "V2G_G2V_Technologies": "bidirectional vehicle grid interaction (grid v2g technology, vehicle g2v, bidirectional energy)",
    "Charging_Infrastructure": "ev charging systems and network (charging infrastructure)",
    "Reactive_Power_Management": "reactive power control in grids (reactive power)",
    "DC_Microgrid": "dc based local energy networks (dc microgrid)"
}
AUTO_TOP_SEEDS_6 = {
    "Traffic_Planning": "urban traffic congestion dynamics (traffic congestion)",
    "Transport_Infrastructure": "urban transport network and infrastructure (transport infrastructure)",
    "Shared_Mobility": "ride sharing and shared transport systems (shared mobility)",
    "Mobility_Demand_Forecasting": "urban travel demand prediction (demand forecasting)",
    "Micro_Mobility": "e scooters bikes and small personal transport (micro mobility)",
    "Node_Activity": "traffic node and intersection activity dynamics (node activity)"
}
AUTO_TOP_SEEDS_7 = {
    "InLine_Quality_Inspection": "inline defect and weld quality inspection (defect detection, weld quality)",
    "Error_Proofing_PokaYoke": "mistake proofing and error prevention (poka yoke)",
    "Predictive_Maintenance": "vehicle and equipment predictive maintenance (vehicle maintenance, predictive maintenance pd)",
    "Process_Monitoring_Optimization": "real time monitoring and waste minimizing stable processes (real time monitoring, proactively finding deviations, minimizing waste optimizing, quality constant process)"
}
AUTO_TOP_SEEDS_8 = {
    "Autonomous_Delivery_Robots": "autonomous robotic delivery and last mile transport (automated delivery, robotic delivery shipping)",
    "Warehouse_Intelligence_Robots": "autonomous warehouse robots for inventory tracking mapping inspection and stock counting (counting stock warehouse)",
    "AGV_Systems": "automated guided vehicles for structured factory and warehouse transport (automated guided vehicle)",
    "Hybrid_Modular_Robotics": "hybrid and modular robotic system architectures combining multiple robot types into reconfigurable platforms (hybrid modular)"
}
AUTO_TOP_SEEDS_9 = {
    "Intrusion_Detection": "network intrusion detection (intrusion detection)",
    "Cyber_Physical_Security": "security of cyber physical automotive systems (cyber physical)",
    "InVehicle_Network_Protocols": "automotive communication and bus protocols (controller area network, protocols)",
    "Cryptography_Key_Management": "encryption ciphering and cryptographic key management (ciphering key)",
    "Integrity_Protection": "data and message integrity protection mechanisms (integrity protection)",
    "Functional_Safety": "automotive functional safety and fail safe systems (functional safety)"
}

# Main dictionary organizing technologies by area
AUTO_TOP_SEEDS = {
    "Perception": AUTO_TOP_SEEDS_1,
    "Communication_Technologies": AUTO_TOP_SEEDS_2,
    "Energy_Source": AUTO_TOP_SEEDS_3,
    "Energy_Storage": AUTO_TOP_SEEDS_4,
    "Energy_Management": AUTO_TOP_SEEDS_5,
    "Urban_Mobility": AUTO_TOP_SEEDS_6,
    "Manufacturing": AUTO_TOP_SEEDS_7,
    "Robotics": AUTO_TOP_SEEDS_8,
    "Cybersecurity": AUTO_TOP_SEEDS_9,
}

# Mapping between display area names and dictionary keys
AREA_NAME_MAPPING = {
    "Sensing Perception VehicleUnderstanding": "Perception",
    "Robotic Factory Autonomous Delivery": "Robotics", 
    "Manufacturing Industrial AI": "Manufacturing",
    "Communication Technologies": "Communication_Technologies",
    "Energy Source": "Energy_Source",
    "Energy Storage": "Energy_Storage", 
    "Energy Management": "Energy_Management",
    "Urban Mobility": "Urban_Mobility",
    "Cybersecurity": "Cybersecurity",
    "Perception": "Perception",
    "Robotics": "Robotics",
    "Manufacturing": "Manufacturing",
    "Communication_Technologies": "Communication_Technologies",
}

def get_correct_paths():
    """Get absolute paths based on folder structure"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    
    rag_components_path = os.path.join(project_root, '03_notebooks', 'rag_notebooks', 'rag_components')
    vector_index_path = os.path.join(project_root, '04_models', 'vector_index')
    
    return rag_components_path, vector_index_path, project_root

def import_your_components():
    """Import FAISS retriever with exact paths"""
    rag_components_path, _, _ = get_correct_paths()
    
    faiss_retriever_path = os.path.join(rag_components_path, 'faiss_retriever.py')
    
    if not os.path.exists(faiss_retriever_path):
        return None, f"FAISS Retriever not found at: {faiss_retriever_path}"
        
    try:
        if rag_components_path not in sys.path:
            sys.path.insert(0, rag_components_path)
        
        spec = importlib.util.spec_from_file_location("faiss_retriever", faiss_retriever_path)
        retriever_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(retriever_module)
        return retriever_module, None
    except Exception as e:
        return None, f"Error importing FAISS retriever: {str(e)}"

def import_predictive_components():
    """Import predictive model components"""
    _, _, project_root = get_correct_paths()

    from pathlib import Path
    notebooks_root = Path(project_root) / "03_notebooks"
    predictive_components_path = notebooks_root / "predictive_notebooks" / "predictive_components"

    if not predictive_components_path.exists():
        return None, "Predictive components directory not found"

    if str(notebooks_root) not in sys.path:
        sys.path.insert(0, str(notebooks_root))

    try:
        from predictive_notebooks.predictive_components import analytics as predictive_analytics

        predictive_functions = {
            "load_area_tech_ts": predictive_analytics.load_area_tech_ts,
            "get_fastest_growing_topics": predictive_analytics.get_fastest_growing_topics,
            "get_likely_to_mature_next_year": predictive_analytics.get_likely_to_mature_next_year,
            "plot_simple_timeseries": predictive_analytics.plot_simple_timeseries,
            "plot_maturity_derivatives": predictive_analytics.plot_maturity_derivatives,
        }

        return predictive_functions, None

    except ImportError as e:
        return None, f"Error importing analytics module: {e}"
    except Exception as e:
        return None, f"Error in import_predictive_components: {e}"
    
def import_query_expander():
    """Import the existing query expander module (if it exists)"""
    rag_components_path, _, _ = get_correct_paths()
    expander_path = os.path.join(rag_components_path, 'query_expander.py')
    
    if not os.path.exists(expander_path):
        return None, None
    
    try:
        if rag_components_path not in sys.path:
            sys.path.insert(0, rag_components_path)
        
        spec = importlib.util.spec_from_file_location("query_expander", expander_path)
        expander_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(expander_module)
        return expander_module, None
    except Exception as e:
        return None, None

def setup_groq_client():
    """Setup Groq client for LLM interactions"""
    try:
        from groq import Groq
        from dotenv import load_dotenv
        load_dotenv()
        
        api_key = os.getenv('GROQ_API_KEY')
        if not api_key:
            return None, "GROQ_API_KEY not found in environment variables"
        
        client = Groq(api_key=api_key)
        return client, None
    except ImportError:
        return None, "Groq package not installed. Run: pip install groq"
    except Exception as e:
        return None, f"Error setting up Groq client: {str(e)}"

def analyze_question_type(question):
    """Analyze question type to determine processing approach"""
    question_lower = question.lower()
    
    # Define question type characteristics
    question_type = {
        'is_startup_query': any(keyword in question_lower for keyword in 
                               ['startup', 'company', 'companies', 'venture', 'business', 'funding']),
        'is_patent_query': any(keyword in question_lower for keyword in 
                              ['patent', 'intellectual property', 'ip', 'jurisdiction', 'ep', 'us', 'wo']),
        'is_research_query': any(keyword in question_lower for keyword in 
                                ['research', 'study', 'paper', 'academic', 'method', 'experiment']),
        'is_trend_query': any(keyword in question_lower for keyword in 
                             ['trend', 'forecast', 'future', 'emerging', 'development', 'innovation', 
                              'pain point', 'challenge', 'barrier', 'obstacle']),
        
        # Predictive model detection - UPDATED for commercial interest
        'is_growth_query': any(keyword in question_lower for keyword in 
                              ['fastest growing', 'growth rate', 'increasing', 'growing technology',
                               'accelerating', 'expanding', 'growth trend', 'rapid growth']),
        
        'is_commercial_interest_query': any(keyword in question_lower for keyword in 
                                ['commercial interest', 'commercial rising', 'market interest',
                                 'business interest', 'industry interest', 'commercial adoption',
                                 'market adoption', 'industry adoption', 'rising commercial']),
        
        'is_technology_query': any(keyword in question_lower for keyword in 
                                  ['technology', 'tech', 'system', 'solution', 'application', 
                                   'algorithm', 'agent', 'agents', 'model', 'framework']),
        
        'is_list_query': any(keyword in question_lower for keyword in 
                            ['list', 'which', 'what are', 'name', 'show me', 'examples']),
        
        'is_summary_query': any(keyword in question_lower for keyword in 
                               ['summarize', 'overview', 'explain', 'describe', 'what is']),
    }
    
    # Determine thresholds based on question type
    if question_type['is_startup_query'] or question_type['is_list_query']:
        semantic_threshold = 0.15
        keyword_threshold = 0.1
        k_semantic = 4
        k_keyword = 2
    elif question_type['is_patent_query']:
        semantic_threshold = 0.25
        keyword_threshold = 0.15
        k_semantic = 4
        k_keyword = 2
    elif question_type['is_research_query']:
        semantic_threshold = 0.3
        keyword_threshold = 0.2
        k_semantic = 3
        k_keyword = 2
    else:
        semantic_threshold = 0.25
        keyword_threshold = 0.15
        k_semantic = 3
        k_keyword = 2
    
    return question_type, semantic_threshold, keyword_threshold, k_semantic, k_keyword

def determine_question_category(question):
    """Route questions to appropriate processor (RAG or predictive)"""
    question_type, _, _, _, _ = analyze_question_type(question)
    
    # Route to predictive model for growth or commercial interest questions
    if question_type['is_growth_query']:
        return 'predictive_growth'
    elif question_type['is_commercial_interest_query']:
        # Check for time indicators (next year, coming year, etc.)
        time_keywords = ['next', 'future', 'coming', 'likely', 'forecast', 'prediction', '12 months', 'next year']
        if any(word in question.lower() for word in time_keywords):
            return 'predictive_commercial_interest'
    
    # Default to RAG for all other questions
    return 'rag_only'

def get_technology_definition(tech_name, area_display):
    """Get technology definition using proper mapping"""
    # Normalize tech name to match dictionary keys
    tech_key = tech_name.replace(' ', '_')
    
    # Map display area name to dictionary key
    area_key = AREA_NAME_MAPPING.get(area_display, area_display.replace(' ', '_'))
    
    # Try to get definition from dictionary
    definition = None
    if area_key in AUTO_TOP_SEEDS:
        # Try exact match first
        if tech_key in AUTO_TOP_SEEDS[area_key]:
            definition = AUTO_TOP_SEEDS[area_key][tech_key]
        else:
            # Try to find a close match
            for dict_key, dict_value in AUTO_TOP_SEEDS[area_key].items():
                if dict_key.lower() == tech_key.lower() or dict_key.replace('_', ' ').lower() == tech_key.replace('_', ' ').lower():
                    definition = dict_value
                    break
    
    return definition

def get_area_display_for_tech(tech_name: str) -> str:
    """Find the area name for a given technology from the dictionary"""
    tech_key = tech_name.replace(" ", "_")
    for area_key, tech_dict in AUTO_TOP_SEEDS.items():
        if tech_key in tech_dict:
            return area_key.replace("_", " ")
    return "Unassigned Area"

def format_predictive_results(results_df, category):
    """Format predictive model results for display"""
    if results_df is None:
        return "Error: Predictive model returned None results."
    
    if hasattr(results_df, 'empty') and results_df.empty:
        return "No predictive insights available for this query (empty results)."
    
    try:
        # Growth query format (academic growth)
        if category == 'predictive_growth':
            formatted = "**Academic Growth in Automotive Technologies**\n\n"
            
            required_cols = ['auto_tech_cluster', 'auto_focus_area', 'growth_slope_n_total']
            missing_cols = [col for col in required_cols if col not in results_df.columns]
            
            if missing_cols:
                return f"Error: Missing required columns in growth data: {missing_cols}"
            
            # Create formatted output for each technology
            for idx, row in enumerate(results_df.head(10).itertuples(), 1):
                tech = row.auto_tech_cluster
                tech_display = tech.replace('_', ' ') if isinstance(tech, str) else str(tech)
                
                area = row.auto_focus_area
                area_display = area.replace('_', ' ') if isinstance(area, str) else str(area)
                
                growth = getattr(row, 'growth_slope_n_total', 0)
                
                definition = get_technology_definition(tech, area_display)
                
                formatted += f"**{idx}. {tech_display}**\n\n"
                if definition:
                    formatted += f"**Definition:** {definition[0].upper() + definition[1:]}\n\n"
                else:
                    formatted += f"**Definition:** Technology for {tech_display.lower()} applications\n\n"
                formatted += f"**Area:** {area_display}\n\n"
                
                if isinstance(growth, (int, float)):
                    growth_pct = growth * 100
                    formatted += f"**Quarterly Growth Rate (Last Quarter vs. Previous Quarter):** {growth_pct:.1f}%\n\n"
                else:
                    formatted += f"**Quarterly Growth Rate (Last Quarter vs. Previous Quarter):** {growth}\n\n"
                
                recent_activity = getattr(row, 'n_total_last', getattr(row, 'recent_activity', 0))
                formatted += f"**Journal Articles Published (Last Quarter):** {int(recent_activity)} documents\n\n"
        
        # Commercial interest query format
        elif category == 'predictive_commercial_interest':
            formatted = "**Rising Commercial Interest in Automotive Technologies**\n\n"
            
            required_cols = ['auto_tech_cluster', 'auto_focus_area', 'last_share_patent', 'forecast_share_patent_mean', 'delta_share_patent']
            missing_cols = [col for col in required_cols if col not in results_df.columns]
            
            if missing_cols:
                return f"Error: Missing required columns in commercial interest data: {missing_cols}"
            
            # Create formatted output for each technology
            for idx, row in enumerate(results_df.head(15).itertuples(), 1):
                tech = row.auto_tech_cluster
                tech_display = tech.replace('_', ' ') if isinstance(tech, str) else str(tech)
                
                area = row.auto_focus_area
                area_display = area.replace('_', ' ') if isinstance(area, str) else str(area)
                
                definition = get_technology_definition(tech, area_display)
                
                formatted += f"**{idx}. {tech_display}**\n\n"
                if definition:
                    formatted += f"**Definition:** {definition[0].upper() + definition[1:]}\n\n"
                else:
                    formatted += f"**Definition:** Technology for {tech_display.lower()} applications\n\n"
                formatted += f"**Area:** {area_display}\n\n"
        
        else:
            formatted = f"**Predictive Insights**\n\n"
            if hasattr(results_df, 'head'):
                formatted += results_df.head(10).to_markdown()
            else:
                formatted += str(results_df)
        
        return formatted
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        return f"Error formatting predictive results: {str(e)}"

def build_smart_prompt(question, context, predictive_insights=None):
    """Build LLM prompt with question-specific guidance"""
    question_lower = question.lower()
    
    # Detect query type for targeted guidance
    is_startup_question = any(keyword in question_lower for keyword in 
                            ['startup', 'company', 'companies', 'venture', 'business', 'funding'])
    
    is_patent_question = any(keyword in question_lower for keyword in 
                           ['patent', 'intellectual property', 'ip', 'jurisdiction', 'ep', 'us', 'wo'])
    
    is_research_question = any(keyword in question_lower for keyword in 
                             ['research', 'study', 'paper', 'academic', 'scientific', 'methodology',
                              'journal', 'article', 'publication'])
    
    is_trend_question = any(keyword in question_lower for keyword in 
                          ['trend', 'forecast', 'future', 'emerging', 'development', 'innovation', 'pain point', 'challenge'])
    
    is_commercial_question = any(keyword in question_lower for keyword in 
                             ['commercial', 'market', 'business', 'industry', 'adoption', 'interest'])
    
    is_technology_question = any(keyword in question_lower for keyword in 
                               ['technology', 'tech', 'system', 'solution', 'application', 'deployment', 'agent', 'agents'])
    
    # Build targeted guidance sections
    guidance_sections = []
    
    # Patent guidance
    if is_patent_question:
        guidance_sections.append("""
🔍 **PATENT QUERY GUIDANCE:**
- Always include patent abstract summary
- Skip patents without abstracts
- Use exact source format: [Source: Lens.org Automotive Technology Patents 2025]
""")
    
    # Startup guidance
    if is_startup_question:
        guidance_sections.append("""
🚀 **STARTUP QUERY GUIDANCE:**
- Extract company names and details
- Use startup-specific source formats
""")
    
    # Research guidance
    if is_research_question:
        guidance_sections.append("""
📚 **RESEARCH QUERY GUIDANCE:**
- Always include short summary of each paper (2-3 sentences)
- Use exact source format: [Source: Lens.org Automotive Research Papers Abstracts 2025]
""")
    
    # Commercial interest guidance
    if is_commercial_question:
        guidance_sections.append("""        
🎯 **COMMERCIAL INTEREST GUIDANCE:**
- Focus on market adoption and business applications
- Note industry partnerships and deployments
- Highlight commercial viability indicators
""")
    
    # Unified referencing requirements
    unified_referencing = """
🔗 **REFERENCING REQUIREMENTS:**
- Patents: [Source: Lens.org Automotive Technology Patents 2025]
- Research Papers: [Source: Lens.org Automotive Research Papers Abstracts 2025]
- Startups: [Source: Seedtable Best Automotive Industry Startups to Watch in 2025] 
  OR [Source: AutoTechInsight Automotive Startup Profiles & Tracker]
- Industry Reports: Use specific report titles
"""
    
    # General guidance
    general_guidance = """
📋 **GENERAL ANSWER GUIDELINES:**
- Be specific and use information from context only
- Use consistent source formatting
- Include abstracts for patents and summaries for research papers
"""
    
    # Combine guidance sections
    targeted_guidance = "\n\n".join(guidance_sections)
    
    prompt = f"""
CONTEXT:
{context}

USER QUESTION:
{question}

ANALYSIS INSTRUCTIONS:
You are an automotive technology intelligence analyst. Provide detailed answers based strictly on the context.

{targeted_guidance}

{unified_referencing}

{general_guidance}

ANSWER STRUCTURE:
1. Direct answer to the main question
2. Supporting details with specific examples
3. Source citations in exact format specified

ANSWER:
"""
    return prompt

def determine_source_count(question):
    """Determine how many sources to retrieve based on question complexity"""
    question_lower = question.lower()
    
    if any(keyword in question_lower for keyword in ['summarize', 'comprehensive', 'overall', 'complete', 'latest']):
        return 5
    elif any(keyword in question_lower for keyword in ['list', 'which', 'what are', 'show all', 'show me']):
        return 5
    elif any(keyword in question_lower for keyword in ['specific', 'exact', 'precise', 'detailed']):
        return 3
    else:
        return 4

def format_source_name(source_file):
    """Format source file names for display"""
    name_mapping = {
        # Automotive Papers
        'a_benchmark_framework_for_AL_models_in_automotive_aerodynamics.txt': 'AI in Automotive Aerodynamics Research',
        'AL_agents_in_engineering_design_a_multiagent_framework_for_aesthetic_and_aerodynamic_car_design.txt': 'AI Agents in Car Design Research',
        'automating_automotive_software_development_a_synergy_of_generative_AL_and_formal_methods.txt': 'AI for Automotive Software Development',
        'automotive-software-and-electronics-2030-full-report.txt': 'Automotive Software 2030 Report',
        'drive_disfluency-rich_synthetic_dialog_data_generation_framework_for_intelligent_vehicle_environments.txt': 'AI Dialogue Systems for Vehicles',
        'Embedded_acoustic_intelligence_for_automotive_systems.txt': 'Acoustic AI for Automotive Systems',
        'enhanced_drift_aware_computer_vision_achitecture_for_autonomous_driving.txt': 'Computer Vision for Autonomous Driving',
        'Gen_AL_in_automotive_applications_challenges_and_opportunities_with_a_case_study_on_in-vehicle_experience.txt': 'Generative AI in Automotive Applications',
        'generative_AL_for_autonomous_driving_a_review.txt': 'Generative AI for Autonomous Driving Review',
        'leveraging_vision_language_models_for_visual_grounding_and_analysis_of_automative_UI.txt': 'Vision-Language Models for Automotive UI',
        
        # Tech Reports
        'bog_ai_value_2025.txt': 'BCG: AI Value Creation 2025',
        'mckinsey_tech_trends_2025.txt': 'McKinsey Technology Trends 2025',
        'wef_emerging_tech_2025.txt': 'WEF: Emerging Technologies 2025',
        
        # Processed Files
        'autotechinsight_startups_processed.txt': 'AutoTechInsight Automotive Startup Profiles & Tracker',
        'seedtable_startups_processed.txt': 'Seedtable Best Automotive Industry Startups to Watch in 2025',
        'automotive_papers_processed.txt': 'Lens.org Automotive Research Papers Abstracts 2025',
        'automotive_patents_processed.txt': 'Lens.org Automotive Technology Patents 2025',
        
        # Generic fallbacks
        'startup': 'Startup Database',
        'patent': 'Patent Database',
        'paper': 'Research Database',
        'report': 'Industry Report',
    }
    
    if source_file in name_mapping:
        return name_mapping[source_file]
    
    source_lower = source_file.lower()
    for key, value in name_mapping.items():
        if key in source_lower:
            return value
    
    return source_file.replace('.txt', '').replace('_', ' ').title()

# Initialize components with lazy loading
@st.cache_resource
def initialize_rag_system():
    """Initialize all RAG components"""
    rag_components_path, vector_index_path, project_root = get_correct_paths()
    
    # Check if vector index exists
    if not os.path.exists(vector_index_path):
        return None, None, None, f"Vector index not found at: {vector_index_path}"
    
    # Check for FAISS files
    faiss_files = ['faiss_index.bin', 'texts.pkl', 'metadata.pkl']
    missing_files = []
    for file in faiss_files:
        if not os.path.exists(os.path.join(vector_index_path, file)):
            missing_files.append(file)
    
    if missing_files:
        return None, None, None, f"FAISS files missing: {', '.join(missing_files)}"
    
    # Import FAISS retriever
    retriever_module, retriever_error = import_your_components()
    if retriever_error:
        return None, None, None, retriever_error
    
    # Import query expander
    expander_module, expander_error = import_query_expander()
    
    # Setup Groq client
    groq_client, groq_error = setup_groq_client()
    if groq_error:
        return None, None, None, groq_error
    
    # Initialize FAISS retriever
    try:
        retriever = retriever_module.FAISSRetriever(vector_index_path)
        
        # Initialize query expander if available
        query_expander = None
        if expander_module and not expander_error:
            try:
                query_expander = expander_module.QueryExpander()
            except Exception as e:
                query_expander = None
        
        return retriever, groq_client, query_expander, None
        
    except Exception as e:
        return None, None, None, f"Error initializing FAISS retriever: {str(e)}"

@st.cache_resource
def initialize_predictive_system():
    """Initialize predictive model components"""
    try:
        predictive_functions, error = import_predictive_components()
        if error:
            return None
        return predictive_functions
    except Exception as e:
        return None

def get_targeted_keyword_queries(question, question_type):
    """Generate targeted keyword queries based on question type"""
    question_lower = question.lower()
    keyword_queries = []
    
    # Startup related keywords
    if question_type['is_startup_query']:
        keyword_queries.extend([
            "automotive startup", "AI company automotive", "autonomous vehicle company",
            "electric vehicle startup", "mobility tech company", "car technology startup"
        ])
    
    # Patent related keywords
    if question_type['is_patent_query']:
        keyword_queries.extend([
            "automotive patent", "AI patent automotive", "vehicle technology patent",
            "patent EP automotive", "patent US automotive", "patent WO automotive"
        ])
    
    # Research related keywords
    if question_type['is_research_query']:
        keyword_queries.extend([
            "automotive research", "AI study automotive", "vehicle technology research",
            "autonomous driving study", "electric vehicle research", "automotive AI paper"
        ])
    
    # Commercial interest related keywords - UPDATED
    if question_type['is_commercial_interest_query']:
        keyword_queries.extend([
            "commercial interest automotive", "market adoption automotive", "industry adoption automotive",
            "commercial deployment automotive", "business applications automotive", "market ready automotive",
            "commercial viability automotive", "industry partnerships automotive", "market growth automotive"
        ])
    
    # Technology related keywords
    if question_type['is_technology_query']:
        # Extract technology terms from question
        tech_terms = re.findall(r'\b[A-Z][a-z]+\b', question)
        for term in tech_terms[:3]:
            if len(term) > 3 and term.lower() not in ['ai', 'automotive', 'vehicle', 'car']:
                keyword_queries.append(f"{term} automotive")
                keyword_queries.append(f"{term} vehicle")
        
        keyword_queries.extend([
            "automotive technology", "vehicle system", "car technology",
            "automotive solution", "vehicle application", "automotive AI system"
        ])
    
    # General automotive/AI keywords
    keyword_queries.extend([
        "automotive AI", "vehicle artificial intelligence", "autonomous vehicle",
        "electric vehicle", "connected car", "smart mobility"
    ])
    
    return list(set(keyword_queries))[:8]

def universal_hybrid_retrieval(question, retriever, query_expander=None, k=4):
    """Hybrid retrieval combining semantic and keyword search"""
    all_results = []
    
    # Identify question type for threshold tuning
    question_type, semantic_threshold, keyword_threshold, k_semantic, k_keyword = analyze_question_type(question)
    
    # Semantic search with query expansion
    if query_expander:
        try:
            expanded_queries = query_expander.expand_query(question, use_llm=False)
            if not expanded_queries:
                expanded_queries = [question]
        except:
            expanded_queries = [question]
    else:
        expanded_queries = [question]
    
    for query in expanded_queries[:3]:
        try:
            semantic_results = retriever.retrieve_with_sources(
                query, 
                k=k_semantic, 
                threshold=semantic_threshold
            )
            all_results.extend(semantic_results)
        except Exception as e:
            continue
    
    # Targeted keyword search
    keyword_queries = get_targeted_keyword_queries(question, question_type)
    
    for keyword_query in keyword_queries[:5]:
        try:
            keyword_results = retriever.retrieve_with_sources(
                keyword_query,
                k=k_keyword,
                threshold=keyword_threshold
            )
            
            # Filter for relevance
            filtered_results = []
            for result in keyword_results:
                content = result.get('text', result.get('content', '')).lower()
                if any(term in content for term in keyword_query.split()[:2]):
                    filtered_results.append(result)
            
            all_results.extend(filtered_results)
        except Exception as e:
            continue
    
    # Remove duplicates
    unique_results = []
    seen_content = set()
    
    for result in all_results:
        content = result.get('text', result.get('content', ''))
        content_start = content[:250]
        source = result.get('source_file', 'unknown')
        signature = f"{source}:{content_start}"
        
        if signature not in seen_content:
            seen_content.add(signature)
            unique_results.append(result)
    
    # Sort by similarity score
    unique_results.sort(key=lambda x: x.get('similarity_score', 0), reverse=True)
    
    return unique_results[:k]

def retrieve_with_expansion(question, retriever, query_expander=None, k=4):
    """Main retrieval function"""
    return universal_hybrid_retrieval(question, retriever, query_expander, k)

def process_predictive_query(question, predictive_functions):
    """Handle questions that require predictive model"""
    if not predictive_functions:
        return {
            'answer': "⚠️ Predictive model components are not available.",
            'sources': [],
            'success': True,
            'source_count': 0,
            'predictive_used': False,
        }

    try:
        ts_data = predictive_functions["load_area_tech_ts"]()
        results = None
        technology_items = []
        view_type = None

        # Determine question type
        category = determine_question_category(question)

        # Academic growth query
        if category == "predictive_growth":
            # Get broad list then filter
            base_results = predictive_functions["get_fastest_growing_topics"](ts_data, top_n=50)

            if base_results is None or base_results.empty:
                return {
                    'answer': "No predictive growth insights available.",
                    'sources': [],
                    'success': True,
                    'source_count': 0,
                    'predictive_used': True,
                    'technology_items': [],
                }

            # Filter for positive growth only
            base_results = base_results[base_results["growth_slope_n_total"] > 0].copy()
            base_results = base_results.sort_values("n_total_last", ascending=False)
            results = base_results.head(5).reset_index(drop=True)
            view_type = "growth"
            technology_items = []

            if not results.empty:
                for idx, row in enumerate(results.itertuples(), 1):
                    tech = row.auto_tech_cluster
                    area = row.auto_focus_area

                    tech_display = tech.replace("_", " ") if isinstance(tech, str) else str(tech)
                    raw_area_display = area.replace("_", " ") if isinstance(area, str) else str(area)

                    # Get area from seed dictionary if not available
                    if (not isinstance(area, str)) or raw_area_display.strip().lower() == "global":
                        area_display = get_area_display_for_tech(tech)
                    else:
                        area_display = raw_area_display

                    definition = get_technology_definition(tech, area_display)
                    growth_pct = row.growth_slope_n_total * 100
                    recent_activity = int(row.n_total_last)

                    text = f"**{idx}. {tech_display}**\n\n"
                    if definition:
                        text += f"**Definition:** {definition[0].upper() + definition[1:]}\n\n"
                    else:
                        text += f"**Definition:** Technology for {tech_display.lower()} applications\n\n"
                    text += f"**Area:** {area_display}\n\n"
                    text += f"**Quarterly Growth Rate (Last Quarter vs. Previous Quarter):** {growth_pct:.1f}% \n\n"
                    text += f"**Papers Published (Last Quarter):** {recent_activity} \n\n"

                    fig = None
                    try:
                        fig = predictive_functions["plot_simple_timeseries"](
                            ts_data,
                            area=area,
                            tech=tech,
                        )
                    except Exception as e:
                        print(f"Graph error: {e}")

                    technology_items.append({
                        "text": text,
                        "figure": fig,
                        "tech_display": tech_display,
                        "area_display": area_display,
                    })

        # Commercial interest query
        elif category == "predictive_commercial_interest":
            # Get broader candidate pool
            base_results = predictive_functions["get_likely_to_mature_next_year"](
                ts_data,
                horizon=12,
                top_n=20,
            )

            view_type = "commercial_interest"
            technology_items = []

            # Filter and sort for positive delta_share_patent
            if base_results is not None and not base_results.empty:
                results = base_results.copy()

                if "delta_share_patent" in results.columns:
                    results = results[results["delta_share_patent"] > 0].copy()
                    results = results.sort_values("delta_share_patent", ascending=False)
                    results = results.head(5).reset_index(drop=True)
                else:
                    results = results.head(5).reset_index(drop=True)
            else:
                results = None

            if results is not None and not results.empty:
                for idx, row in enumerate(results.itertuples(), 1):
                    tech = row.auto_tech_cluster
                    area = row.auto_focus_area

                    tech_display = tech.replace("_", " ") if isinstance(tech, str) else str(tech)
                    area_display = get_area_display_for_tech(tech)
                    definition = get_technology_definition(tech, area_display)

                    text = f"**{idx}. {tech_display}**\n\n"
                    if definition:
                        text += f"**Definition:** {definition[0].upper() + definition[1:]}\n\n"
                    else:
                        text += f"**Definition:** Technology for {tech_display.lower()} applications\n\n"
                    text += f"**Area:** {area_display}\n\n"

                    # Graph for commercial interest
                    fig = None
                    if "plot_maturity_derivatives" in predictive_functions:
                        try:
                            fig = predictive_functions["plot_maturity_derivatives"](
                                ts_data,
                                area=area,
                                tech=tech,
                            )
                        except Exception as e:
                            print(f"Warning: Could not generate commercial interest graph for {tech}: {e}")
                            fig = None

                    technology_items.append(
                        {
                            "text": text,
                            "figure": fig,
                            "tech_display": tech_display,
                            "area_display": area_display,
                        }
                    )

        else:
            return {
                'answer': "This question doesn't match any predictive model categories.",
                'sources': [],
                'success': True,
                'source_count': 0,
                'predictive_used': False,
            }

        # Set title based on view type
        if view_type == "growth":
            insights_title = "Academic Growth in Automotive Technologies"
        else:
            insights_title = "Rising Commercial Interest in Automotive Technologies"

        answer = f"""
##### {insights_title}

**Methodology Note**  
- Based on time-series analysis of automotive technology publications and patents from Lens.org  
- Forecasts derived from historical growth and patent dynamics  
- Updated with the latest available data
- Only showing quarters where data was available

"""

        return {
            'answer': answer,
            'sources': [],
            'success': True,
            'source_count': 0,
            'predictive_used': True,
            'predictive_results': results.to_dict('records') if hasattr(results, 'to_dict') else [],
            'technology_items': technology_items,
            'view_type': view_type,
        }

    except Exception as e:
        return {
            'answer': f"Error processing predictive query: {str(e)}",
            'sources': [],
            'success': False,
            'predictive_used': True,
            'technology_items': [],
        }

def process_rag_query(question, retriever, groq_client, query_expander=None):
    """Handle RAG-based questions"""
    try:
        k = determine_source_count(question)
        
        # Retrieve relevant documents
        retrieved_data = retrieve_with_expansion(
            question, 
            retriever, 
            query_expander=query_expander, 
            k=k
        )
        
        if not retrieved_data:
            return {
                'answer': "I couldn't find relevant information in our knowledge base for this specific question.",
                'sources': [],
                'success': True,
                'source_count': k,
                'predictive_used': False,
                'graphs': []
            }
        
        # Build context from retrieved data
        context_parts = []
        for i, item in enumerate(retrieved_data):
            content = item.get('text', item.get('content', ''))
            source_file = item.get('source_file', 'unknown')
            
            readable_name = format_source_name(source_file)
            similarity = item.get('similarity_score', 0)
            
            context_parts.append(f"--- DOCUMENT {i+1} ---")
            context_parts.append(f"Source: {readable_name}")
            context_parts.append(f"Relevance Score: {similarity:.3f}")
            context_parts.append(f"Content:\n{content}")
            context_parts.append("")
        
        context = "\n".join(context_parts)
        prompt = build_smart_prompt(question, context)
        
        # Generate answer using LLM
        max_tokens = 800 if len(context) > 3000 else 600
        
        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0.3
        )
        
        answer = response.choices[0].message.content
        
        return {
            'answer': answer,
            'sources': retrieved_data,
            'success': True,
            'source_count': k,
            'predictive_used': False,
            'graphs': []
        }
        
    except Exception as e:
        return {
            'answer': f"I encountered an error while processing your question: {str(e)}",
            'sources': [],
            'success': False,
            'predictive_used': False,
            'graphs': []
        }

def ask_question(question, retriever, groq_client, predictive_functions=None, query_expander=None):
    """Main question processing with routing logic"""
    
    category = determine_question_category(question)
    
    # Route to predictive model for growth or commercial interest
    if category in ['predictive_growth', 'predictive_commercial_interest']:
        if predictive_functions:
            return process_predictive_query(question, predictive_functions)
        else:
            # Fall back to RAG if predictive not available
            return process_rag_query(question, retriever, groq_client, query_expander)
    else:
        # Everything else goes to RAG
        return process_rag_query(question, retriever, groq_client, query_expander)


def main():
    st.set_page_config(
        page_title="INNOVATION INTELLIGENCE SUITE", 
        page_icon="🚗", 
        layout="wide"
    )
    
    st.title("Uncover What's Next in Auto Tech")
    st.markdown("Developed as a Data Science + AI Bootcamp capstone project")
    
    # Initialize RAG system
    if 'rag_initialized' not in st.session_state:
        with st.spinner("Loading RAG system..."):
            st.session_state.retriever, st.session_state.groq_client, st.session_state.query_expander, error = initialize_rag_system()
            
            if error:
                st.session_state.rag_initialized = False
                st.error(f"❌ RAG system initialization failed")
                
                # Show appropriate error messages
                if "Groq package not installed" in error:
                    st.error("**Missing Dependency**")
                    st.info("Install the Groq package: `pip install groq`")
                elif "GROQ_API_KEY" in error:
                    st.error("**API Key Missing**")
                    st.info("Add your Groq API key to the `.env` file")
                elif "Vector index not found" in error or "FAISS files missing" in error:
                    st.error("**Knowledge Base Missing**")
                    st.info("Generate the FAISS vector index first")
                elif "FAISS Retriever not found" in error:
                    st.error("**FAISS Retriever Missing**")
                    st.info("Make sure `faiss_retriever.py` exists in `rag_components/`")
                else:
                    st.error(f"**Error:** {error}")
                    
            elif st.session_state.retriever and st.session_state.groq_client:
                st.session_state.rag_initialized = True
            else:
                st.session_state.rag_initialized = False
                st.error("❌ Failed to initialize RAG system")
    
    # Initialize predictive system
    if 'predictive_initialized' not in st.session_state:
        with st.spinner("Loading predictive model..."):
            st.session_state.predictive_functions = initialize_predictive_system()
            if st.session_state.predictive_functions:
                st.session_state.predictive_initialized = True
            else:
                st.session_state.predictive_initialized = False
                st.warning("⚠️ Predictive model not available - using RAG only")
    
    # Only show query interface if RAG system is initialized
    if not st.session_state.get('rag_initialized', False):
        st.warning("Please fix the initialization issues above to use the system.")
        st.markdown("---")
        st.caption(f"Powered by Innovation Intelligence Suite (2025)")
        return
    
    # Initialize question input in session state
    if 'question_input' not in st.session_state:
        st.session_state.question_input = ""
    
    # Pre-defined button questions - UPDATED with commercial interest question
    button_questions = {
        'research_clicked': "Summarize the latest AI research on autonomous driving vehicles.",
        'patents_clicked': "Show me recent patents on AI for automotive vehicles.",
        'startups_clicked': "Which startups work on automotive and autonomous driving?",
        'trends_clicked': "What are the key challenges and pain points in automotive AI adoption?",
        'agents_clicked': "Summarize latest tech trends in development of AI agents.",
        'growth_clicked': "What are the fastest growing automotive technologies in academia?",
        'commercial_clicked': "For which automotive technologies is the commercial interest rising in the next year?"
    }
    
    # Initialize button flags
    for flag in button_questions.keys():
        if flag not in st.session_state:
            st.session_state[flag] = False
    
    # Check for button clicks
    for flag, question_text in button_questions.items():
        if st.session_state[flag]:
            st.session_state.question_input = question_text
            st.session_state[flag] = False
            st.rerun()
    
    # Query input
    question = st.text_input(
        "💬 Your question:",
        placeholder="e.g., Which startups work on AI for automotive?",
        key="question_input"
    )
    
    # Pre-defined query buttons - UPDATED labels
    st.subheader("Example Questions")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**🚀 Innovation Intelligence**")
        if st.button("Latest AI Research", use_container_width=True, key="research_btn"):
            st.session_state.research_clicked = True
            st.rerun()
        if st.button("Automotive Patents", use_container_width=True, key="patents_btn"):
            st.session_state.patents_clicked = True
            st.rerun()
        if st.button("Startups in Automotive", use_container_width=True, key="startups_btn"):
            st.session_state.startups_clicked = True
            st.rerun()
    
    with col2:
        st.markdown("**📈 Market Insights**")
        if st.button("Industry Pain Points", use_container_width=True, key="trends_btn"):
            st.session_state.trends_clicked = True
            st.rerun()
        if st.button("AI Agents Development", use_container_width=True, key="agents_btn"):
            st.session_state.agents_clicked = True
            st.rerun()
    
    with col3:
        st.markdown("**🔮 Predictive Analytics**")
        if st.button("Academic growth", use_container_width=True, key="growth_btn"):
            st.session_state.growth_clicked = True
            st.rerun()
        if st.button("Commercial interest", use_container_width=True, key="commercial_btn"):
            st.session_state.commercial_clicked = True
            st.rerun()
    
    # Process question
    if question:
        # Determine processing type
        category = determine_question_category(question)
        
        if category.startswith('predictive_'):
            status_msg = "🔮 Running predictive analysis..."
        else:
            status_msg = "🔍 Searching documents and generating answer..."
        
        with st.spinner(status_msg):
            result = ask_question(
                question, 
                st.session_state.retriever, 
                st.session_state.groq_client,
                predictive_functions=st.session_state.predictive_functions,
                query_expander=st.session_state.query_expander
            )
        
        # Display results
        st.subheader("📝 **Answer**")
        
        if result.get('predictive_used', False) and 'technology_items' in result and result['technology_items']:
            view_type = result.get("view_type", "growth")

            if view_type == "growth":
                st.markdown("##### Academic Growth in Automotive Technologies")
            else:
                st.markdown("##### Rising Commercial Interest in Automotive Technologies")

            # Display each technology with its graph
            for item in result['technology_items']:
                tech_container = st.container()

                with tech_container:
                    # Adjust column ratios based on view type
                    if view_type == "growth":
                        col1, col2 = st.columns([2, 1])
                    else:
                        col1, col2 = st.columns([1.2, 1.8])

                    with col1:
                        st.markdown(item['text'])

                    with col2:
                        fig = item.get('figure')
                        if fig is not None:
                            # Adjust graph size for commercial interest view
                            if view_type == "commercial_interest":
                                try:
                                    fig.set_size_inches(10, 4)
                                    fig.subplots_adjust(wspace=0.35)
                                except Exception:
                                    pass

                            title_suffix = "Growth Trend" if view_type == "growth" else "Commercial Interest"
                            st.markdown(
                                f"<div style='text-align: center; margin-bottom: 0.5rem;'><b>{item['tech_display']} - {title_suffix}</b></div>",
                                unsafe_allow_html=True,
                            )
                            st.pyplot(fig, use_container_width=True)

                st.markdown("---")

            # Methodology note
            st.markdown("**Methodology Note**")
            st.markdown("- Based on time-series analysis of automotive technology publications and patents from Lens.org")
            st.markdown("- Forecasts derived from historical growth and patent dynamics")
            st.markdown("- Updated with the latest available data")
            st.markdown("- Only showing quarters where data was available")

            st.caption("*Based on Time-Series Predictive Modelling*")

        elif result.get('predictive_used', False):
            # Predictive used but no technology_items
            st.markdown(result['answer'])
            st.caption("*Based on Time-Series Predictive Modelling*")

        else:
            # RAG queries
            answer_lines = result['answer'].split('\n')
            if len(answer_lines) > 0:
                first_line = answer_lines[0].strip()
                first_line = first_line.replace(':', '').replace('**', '')
                answer_lines[0] = f"##### {first_line}"
                result['answer'] = '\n'.join(answer_lines)
            
            st.markdown(result['answer'])
            
            if result['sources']:
                st.caption(f"*Based on {len(result['sources'])} documents*")
        
        # Display sources for RAG queries
        if result['sources']:
            with st.expander(f"📚 Source Documents ({len(result['sources'])})"):
                for i, source in enumerate(result['sources']):
                    readable_name = format_source_name(source['source_file'])
                    similarity = source.get('similarity_score', 0)
                    
                    st.markdown(f"**{readable_name}** (Relevance: {similarity:.3f})")
                    content = source.get('text', source.get('content', ''))
                    st.text(content[:500] + "..." if len(content) > 500 else content)
                    st.markdown("---")
    
    # Footer
    st.markdown("---")
    st.caption(f"Powered by Innovation Intelligence Suite (2025)")

if __name__ == "__main__":
    main()