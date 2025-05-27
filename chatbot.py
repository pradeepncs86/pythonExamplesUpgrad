"""
Disha Boat - OCP Migration Status Chatbot
Google ADK (Agent Development Kit) implementation with FAISS and Jira integration
"""

import os
import json
import asyncio
import logging
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass
from datetime import datetime

# Google ADK imports
from google_adk import Agent, Tool, LLM
from google_adk.core import AgentConfig, ToolConfig
from google_adk.llm import VertexAILLM
from google_adk.tools import BaseTool
from google_adk.agents import ConversationalAgent

# FAISS for similarity search
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Jira integration
from jira import JIRA
import requests

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class MigrationStatus:
    """Data class for OCP migration status"""
    app_id: str
    app_name: str
    migration_stage: str
    completion_percentage: float
    blockers: List[str]
    last_updated: datetime
    assigned_team: str
    jira_tickets: List[str]

class FAISSEmbeddingStore:
    """FAISS-based embedding store for similarity search"""
    
    def __init__(self, embedding_model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(embedding_model_name)
        self.index = None
        self.documents = []
        self.embeddings = []
        
    def add_documents(self, documents: List[Dict[str, Any]]):
        """Add documents to the embedding store"""
        texts = [doc.get('text', '') for doc in documents]
        embeddings = self.model.encode(texts)
        
        if self.index is None:
            dimension = embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dimension)  # Inner Product for similarity
            
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings.astype('float32'))
        
        self.documents.extend(documents)
        self.embeddings.extend(embeddings)
        
    def similarity_search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Perform similarity search"""
        if self.index is None or len(self.documents) == 0:
            return []
            
        query_embedding = self.model.encode([query])
        faiss.normalize_L2(query_embedding)
        
        scores, indices = self.index.search(query_embedding.astype('float32'), k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.documents):
                result = self.documents[idx].copy()
                result['similarity_score'] = float(score)
                results.append(result)
                
        return results

class IntakeAssessmentTool(BaseTool):
    """Google ADK Tool for intake assessment using FAISS similarity search"""
    
    def __init__(self):
        config = ToolConfig(
            name="intake_assessment",
            description="Assess application readiness for OCP migration using similarity search",
            input_schema={
                "type": "object",
                "properties": {
                    "app_description": {
                        "type": "string",
                        "description": "Description of the application to assess"
                    }
                },
                "required": ["app_description"]
            }
        )
        super().__init__(config)
        
        self.embedding_store = FAISSEmbeddingStore()
        self._initialize_assessment_data()
        
    def _initialize_assessment_data(self):
        """Initialize with sample assessment criteria"""
        assessment_data = [
            {
                "text": "Java Spring Boot application with microservices architecture",
                "readiness_score": 0.9,
                "migration_effort": "Low",
                "recommended_approach": "Containerize and deploy with minimal changes"
            },
            {
                "text": "Legacy monolithic application with database dependencies",
                "readiness_score": 0.4,
                "migration_effort": "High",
                "recommended_approach": "Refactor into microservices before migration"
            },
            {
                "text": "Cloud-native application with REST APIs and stateless design",
                "readiness_score": 0.95,
                "migration_effort": "Very Low",
                "recommended_approach": "Direct migration with configuration updates"
            },
            {
                "text": "Application with file system dependencies and local storage",
                "readiness_score": 0.3,
                "migration_effort": "Very High",
                "recommended_approach": "Redesign storage architecture for cloud compatibility"
            },
            {
                "text": "Node.js application with Express framework and MongoDB",
                "readiness_score": 0.8,
                "migration_effort": "Low",
                "recommended_approach": "Containerize with persistent volume for database"
            },
            {
                "text": "Python Django application with PostgreSQL database",
                "readiness_score": 0.85,
                "migration_effort": "Low",
                "recommended_approach": "Use managed database service and container deployment"
            }
        ]
        self.embedding_store.add_documents(assessment_data)
        
    async def execute(self, **kwargs) -> Dict[str, Any]:
        """Execute intake assessment"""
        try:
            app_description = kwargs.get('app_description', '')
            if not app_description:
                return {
                    "status": "error",
                    "message": "Application description is required"
                }
                
            similar_cases = self.embedding_store.similarity_search(app_description, k=3)
            
            if not similar_cases:
                return {
                    "status": "error",
                    "message": "No similar cases found for assessment"
                }
                
            # Calculate weighted assessment based on similarity
            total_score = 0
            total_weight = 0
            migration_efforts = []
            recommendations = []
            
            for case in similar_cases:
                weight = case['similarity_score']
                total_score += case['readiness_score'] * weight
                total_weight += weight
                migration_efforts.append(case['migration_effort'])
                recommendations.append(case['recommended_approach'])
                
            final_score = total_score / total_weight if total_weight > 0 else 0
            
            return {
                "status": "success",
                "tool": "intake_assessment",
                "assessment": {
                    "readiness_score": round(final_score, 2),
                    "migration_effort": max(set(migration_efforts), key=migration_efforts.count),
                    "recommendations": recommendations[:2],  # Top 2 recommendations
                    "similar_cases": len(similar_cases),
                    "confidence": round(sum(case['similarity_score'] for case in similar_cases) / len(similar_cases), 2)
                },
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error in intake assessment: {str(e)}")
            return {"status": "error", "message": str(e)}

class HardGateTool(BaseTool):
    """Google ADK Tool for hard gate checks using FAISS similarity search"""
    
    def __init__(self):
        config = ToolConfig(
            name="hard_gate_check",
            description="Perform hard gate validation checks for OCP migration",
            input_schema={
                "type": "object",
                "properties": {
                    "check_type": {
                        "type": "string",
                        "description": "Type of gate check to perform"
                    },
                    "app_id": {
                        "type": "string",
                        "description": "Application identifier"
                    }
                },
                "required": ["check_type", "app_id"]
            }
        )
        super().__init__(config)
        
        self.embedding_store = FAISSEmbeddingStore()
        self._initialize_gate_criteria()
        
    def _initialize_gate_criteria(self):
        """Initialize hard gate criteria"""
        gate_criteria = [
            {
                "text": "Security compliance and vulnerability scanning completed",
                "gate_type": "security",
                "mandatory": True,
                "validation_steps": ["SAST scan", "DAST scan", "Dependency check", "Container image scan"],
                "typical_duration": "2-3 days"
            },
            {
                "text": "Performance testing and load testing validation",
                "gate_type": "performance",
                "mandatory": True,
                "validation_steps": ["Load test", "Stress test", "Capacity planning", "Resource optimization"],
                "typical_duration": "3-5 days"
            },
            {
                "text": "Database migration and data integrity verification",
                "gate_type": "data",
                "mandatory": True,
                "validation_steps": ["Data backup", "Migration test", "Rollback plan", "Data validation"],
                "typical_duration": "1-2 days"
            },
            {
                "text": "Network connectivity and firewall configuration",
                "gate_type": "network",
                "mandatory": True,
                "validation_steps": ["Network topology", "Firewall rules", "DNS configuration", "SSL certificates"],
                "typical_duration": "1 day"
            },
            {
                "text": "Application configuration and environment setup",
                "gate_type": "configuration",
                "mandatory": True,
                "validation_steps": ["Config maps", "Secrets management", "Environment variables", "Service discovery"],
                "typical_duration": "1-2 days"
            }
        ]
        self.embedding_store.add_documents(gate_criteria)
        
    async def execute(self, **kwargs) -> Dict[str, Any]:
        """Execute hard gate validation"""
        try:
            check_type = kwargs.get('check_type', '')
            app_id = kwargs.get('app_id', '')
            
            if not check_type or not app_id:
                return {
                    "status": "error",
                    "message": "Both check_type and app_id are required"
                }
                
            relevant_gates = self.embedding_store.similarity_search(check_type, k=5)
            
            validation_results = []
            passed_gates = 0
            total_gates = len(relevant_gates)
            
            for gate in relevant_gates:
                # Simulate gate validation (in real implementation, this would call actual validation services)
                gate_passed = np.random.choice([True, False], p=[0.8, 0.2])  # 80% pass rate simulation
                
                validation_results.append({
                    "gate_type": gate.get('gate_type', 'unknown'),
                    "mandatory": gate.get('mandatory', False),
                    "passed": gate_passed,
                    "validation_steps": gate.get('validation_steps', []),
                    "typical_duration": gate.get('typical_duration', 'Unknown'),
                    "similarity_score": gate['similarity_score']
                })
                
                if gate_passed:
                    passed_gates += 1
                    
            overall_status = "PASSED" if passed_gates == total_gates else "FAILED"
            
            return {
                "status": "success",
                "tool": "hard_gate_check",
                "app_id": app_id,
                "overall_status": overall_status,
                "passed_gates": passed_gates,
                "total_gates": total_gates,
                "validation_results": validation_results,
                "recommendations": self._generate_gate_recommendations(validation_results),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error in hard gate check: {str(e)}")
            return {"status": "error", "message": str(e)}
            
    def _generate_gate_recommendations(self, results: List[Dict]) -> List[str]:
        """Generate recommendations based on gate results"""
        recommendations = []
        failed_gates = [r for r in results if not r['passed']]
        
        if failed_gates:
            recommendations.append("Address failed gate validations before proceeding")
            for gate in failed_gates:
                recommendations.append(f"Complete {gate['gate_type']} validation steps: {', '.join(gate['validation_steps'][:2])}")
        else:
            recommendations.append("All gates passed - ready to proceed with migration")
            
        return recommendations

class AppDataTool(BaseTool):
    """Google ADK Tool for application data analysis using FAISS similarity search"""
    
    def __init__(self):
        config = ToolConfig(
            name="app_data_analysis",
            description="Analyze application data patterns and migration requirements",
            input_schema={
                "type": "object",
                "properties": {
                    "app_description": {
                        "type": "string",
                        "description": "Description of the application"
                    },
                    "data_requirements": {
                        "type": "string",
                        "description": "Data requirements and characteristics"
                    }
                },
                "required": ["app_description"]
            }
        )
        super().__init__(config)
        
        self.embedding_store = FAISSEmbeddingStore()
        self._initialize_app_patterns()
        
    def _initialize_app_patterns(self):
        """Initialize application data patterns"""
        app_patterns = [
            {
                "text": "High-volume transactional application with ACID requirements",
                "data_pattern": "OLTP",
                "storage_recommendation": "Persistent volumes with high IOPS SSD storage",
                "backup_strategy": "Point-in-time recovery with WAL archiving",
                "complexity": "High",
                "estimated_downtime": "2-4 hours",
                "storage_size": "Large (>100GB)"
            },
            {
                "text": "Analytics application with batch processing and reporting",
                "data_pattern": "OLAP",
                "storage_recommendation": "Object storage with data lake architecture",
                "backup_strategy": "Scheduled full backups with compression",
                "complexity": "Medium",
                "estimated_downtime": "4-8 hours",
                "storage_size": "Very Large (>1TB)"
            },
            {
                "text": "Stateless web application with session management",
                "data_pattern": "Session-based",
                "storage_recommendation": "Redis cluster for session storage",
                "backup_strategy": "Configuration backup only",
                "complexity": "Low",
                "estimated_downtime": "30 minutes",
                "storage_size": "Small (<10GB)"
            },
            {
                "text": "File processing application with large binary data",
                "data_pattern": "File-based",
                "storage_recommendation": "Network-attached storage with high throughput",
                "backup_strategy": "Incremental backup with deduplication",
                "complexity": "Medium",
                "estimated_downtime": "1-2 hours",
                "storage_size": "Large (>100GB)"
            },
            {
                "text": "Real-time streaming application with event processing",
                "data_pattern": "Stream-processing",
                "storage_recommendation": "Kafka clusters with persistent volumes",
                "backup_strategy": "Topic replication and offset management",
                "complexity": "High",
                "estimated_downtime": "1 hour",
                "storage_size": "Medium (10-100GB)"
            },
            {
                "text": "Content management system with media files",
                "data_pattern": "Content-based",
                "storage_recommendation": "Object storage with CDN integration",
                "backup_strategy": "Multi-region replication with versioning",
                "complexity": "Medium",
                "estimated_downtime": "2-3 hours",
                "storage_size": "Large (>100GB)"
            }
        ]
        self.embedding_store.add_documents(app_patterns)
        
    async def execute(self, **kwargs) -> Dict[str, Any]:
        """Execute app data analysis"""
        try:
            app_description = kwargs.get('app_description', '')
            data_requirements = kwargs.get('data_requirements', '')
            
            if not app_description:
                return {
                    "status": "error",
                    "message": "Application description is required"
                }
                
            query = f"{app_description} {data_requirements}".strip()
            similar_patterns = self.embedding_store.similarity_search(query, k=3)
            
            if not similar_patterns:
                return {
                    "status": "error",
                    "message": "No matching data patterns found"
                }
                
            # Aggregate recommendations based on similarity
            recommendations = {
                "primary_storage": similar_patterns[0].get('storage_recommendation', ''),
                "backup_strategy": similar_patterns[0].get('backup_strategy', ''),
                "complexity_level": similar_patterns[0].get('complexity', 'Unknown'),
                "estimated_downtime": similar_patterns[0].get('estimated_downtime', 'Unknown'),
                "storage_size_category": similar_patterns[0].get('storage_size', 'Unknown'),
                "data_pattern": similar_patterns[0].get('data_pattern', 'Unknown')
            }
            
            # Additional recommendations based on all matches
            alternative_options = []
            for i, pattern in enumerate(similar_patterns[1:], 1):
                alternative_options.append({
                    "rank": i + 1,
                    "storage_option": pattern.get('storage_recommendation', ''),
                    "confidence": round(pattern['similarity_score'], 2),
                    "complexity": pattern.get('complexity', 'Unknown')
                })
                
            return {
                "status": "success",
                "tool": "app_data_analysis",
                "primary_recommendations": recommendations,
                "alternative_options": alternative_options,
                "matched_patterns": len(similar_patterns),
                "confidence_score": round(similar_patterns[0]['similarity_score'], 2),
                "analysis_timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error in app data analysis: {str(e)}")
            return {"status": "error", "message": str(e)}

class JiraTool(BaseTool):
    """Google ADK Tool for Jira integration using MCP server for real-time lists"""
    
    def __init__(self, jira_url: str = None, username: str = None, api_token: str = None):
        config = ToolConfig(
            name="jira_integration",
            description="Retrieve real-time Jira ticket information for migration tracking",
            input_schema={
                "type": "object",
                "properties": {
                    "app_id": {
                        "type": "string",
                        "description": "Application identifier"
                    },
                    "ticket_type": {
                        "type": "string",
                        "description": "Type of tickets to retrieve (Bug, Story, Task, or all)",
                        "default": "all"
                    }
                },
                "required": ["app_id"]
            }
        )
        super().__init__(config)
        
        self.jira_url = jira_url
        self.jira = None
        
        if jira_url and username and api_token:
            try:
                self.jira = JIRA(server=jira_url, basic_auth=(username, api_token))
            except Exception as e:
                logger.error(f"Failed to initialize Jira connection: {str(e)}")
        
    async def execute(self, **kwargs) -> Dict[str, Any]:
        """Execute Jira ticket retrieval"""
        try:
            app_id = kwargs.get('app_id', '')
            ticket_type = kwargs.get('ticket_type', 'all')
            
            if not app_id:
                return {
                    "status": "error",
                    "message": "Application ID is required"
                }
                
            if not self.jira:
                # Return mock data if Jira is not configured
                return self._generate_mock_jira_data(app_id, ticket_type)
                
            # Build JQL query based on app_id and ticket type
            jql_base = f'project = "OCP-MIGRATION" AND labels = "{app_id}"'
            
            if ticket_type != "all":
                jql_base += f' AND issuetype = "{ticket_type}"'
                
            issues = self.jira.search_issues(jql_base, maxResults=50)
            
            tickets = []
            for issue in issues:
                tickets.append({
                    "key": issue.key,
                    "summary": issue.fields.summary,
                    "status": issue.fields.status.name,
                    "priority": issue.fields.priority.name if issue.fields.priority else "None",
                    "assignee": issue.fields.assignee.displayName if issue.fields.assignee else "Unassigned",
                    "created": issue.fields.created,
                    "updated": issue.fields.updated,
                    "issue_type": issue.fields.issuetype.name,
                    "labels": issue.fields.labels
                })
                
            # Calculate statistics
            status_counts = {}
            priority_counts = {}
            
            for ticket in tickets:
                status = ticket["status"]
                priority = ticket["priority"]
                
                status_counts[status] = status_counts.get(status, 0) + 1
                priority_counts[priority] = priority_counts.get(priority, 0) + 1
                
            return {
                "status": "success",
                "tool": "jira_integration",
                "app_id": app_id,
                "total_tickets": len(tickets),
                "tickets": tickets,
                "statistics": {
                    "status_distribution": status_counts,
                    "priority_distribution": priority_counts
                },
                "query_timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error in Jira integration: {str(e)}")
            return {"status": "error", "message": str(e)}
            
    def _generate_mock_jira_data(self, app_id: str, ticket_type: str) -> Dict[str, Any]:
        """Generate mock Jira data for demonstration"""
        mock_tickets = [
            {
                "key": f"OCP-001",
                "summary": f"Migration assessment for {app_id}",
                "status": "In Progress",
                "priority": "High",
                "assignee": "John Doe",
                "created": "2024-01-15T10:00:00Z",
                "updated": "2024-01-20T14:30:00Z",
                "issue_type": "Story",
                "labels": [app_id, "migration", "assessment"]
            },
            {
                "key": f"OCP-002",
                "summary": f"Security gate validation for {app_id}",
                "status": "To Do",
                "priority": "High",
                "assignee": "Jane Smith",
                "created": "2024-01-16T09:00:00Z",
                "updated": "2024-01-16T09:00:00Z",
                "issue_type": "Task",
                "labels": [app_id, "security", "gate"]
            },
            {
                "key": f"OCP-003",
                "summary": f"Performance testing issue for {app_id}",
                "status": "Done",
                "priority": "Medium",
                "assignee": "Bob Johnson",
                "created": "2024-01-10T08:00:00Z",
                "updated": "2024-01-18T16:00:00Z",
                "issue_type": "Bug",
                "labels": [app_id, "performance", "testing"]
            }
        ]
        
        # Filter by ticket type if specified
        if ticket_type != "all":
            mock_tickets = [t for t in mock_tickets if t["issue_type"] == ticket_type]
            
        # Calculate statistics
        status_counts = {}
        priority_counts = {}
        
        for ticket in mock_tickets:
            status = ticket["status"]
            priority = ticket["priority"]
            
            status_counts[status] = status_counts.get(status, 0) + 1
            priority_counts[priority] = priority_counts.get(priority, 0) + 1
            
        return {
            "status": "success",
            "tool": "jira_integration",
            "app_id": app_id,
            "total_tickets": len(mock_tickets),
            "tickets": mock_tickets,
            "statistics": {
                "status_distribution": status_counts,
                "priority_distribution": priority_counts
            },
            "note": "Mock data - Jira not configured",
            "query_timestamp": datetime.now().isoformat()
        }

class DishaOCPAgent:
    """Main Disha OCP Migration Status Agent using Google ADK"""
    
    def __init__(self, project_id: str, location: str = "us-central1"):
        self.project_id = project_id
        self.location = location
        
        # Initialize LLM using Google ADK
        self.llm = VertexAILLM(
            project_id=project_id,
            location=location,
            model_name="gemini-pro"
        )
        
        # Initialize tools
        self.intake_tool = IntakeAssessmentTool()
        self.hardgate_tool = HardGateTool()
        self.appdata_tool = AppDataTool()
        self.jira_tool = JiraTool()  # Will use mock data unless configured
        
        # Create agent configuration
        agent_config = AgentConfig(
            name="disha_ocp_agent",
            description="AI agent for OCP migration status checks and guidance",
            llm=self.llm,
            tools=[
                self.intake_tool,
                self.hardgate_tool,
                self.appdata_tool,
                self.jira_tool
            ],
            system_message=self._get_system_message()
        )
        
        # Initialize the conversational agent
        self.agent = ConversationalAgent(agent_config)
        
        logger.info("Disha OCP Agent initialized successfully")
        
    def _get_system_message(self) -> str:
        """Get the system message for the agent"""
        return """
        You are Disha, an AI assistant specialized in OpenShift Container Platform (OCP) migration.
        
        Your role is to help development teams assess, plan, and track their application migrations to OpenShift.
        
        You have access to the following tools:
        1. intake_assessment - Assess application readiness for OCP migration
        2. hard_gate_check - Perform validation checks for migration gates
        3. app_data_analysis - Analyze application data patterns and requirements
        4. jira_integration - Retrieve and track Jira tickets for migration projects
        
        Key capabilities:
        - Provide migration readiness assessments
        - Validate compliance and security requirements
        - Recommend storage and data architecture solutions
        - Track migration progress through Jira integration
        - Offer guidance on best practices for OCP migration
        
        Always be helpful, accurate, and provide actionable recommendations.
        When using tools, explain the results clearly and suggest next steps.
        """
        
    def initialize_jira_tool(self, jira_url: str, username: str, api_token: str):
        """Initialize Jira tool with credentials"""
        self.jira_tool = JiraTool(jira_url, username, api_token)
        # Update agent with new tool
        self.agent.config.tools = [
            self.intake_tool,
            self.hardgate_tool,
            self.appdata_tool,
            self.jira_tool
        ]
        logger.info("Jira tool initialized with credentials")
        
    async def chat(self, message: str, app_id: str = None) -> str:
        """Process user message and return response"""
        try:
            # Add app_id context if provided
            if app_id:
                message = f"[App ID: {app_id}] {message}"
                
            response = await self.agent.run(message)
            return response
            
        except Exception as e:
            logger.error(f"Error in chat processing: {str(e)}")
            return f"I apologize, but I encountered an error processing your request: {str(e)}"
            
    async def get_migration_summary(self, app_id: str) -> Dict[str, Any]:
        """Get comprehensive migration status summary for an application"""
        try:
            # Use multiple tools to gather comprehensive information
            summary_data = {}
            
            # Get Jira tickets
            jira_result = await self.jira_tool.execute(app_id=app_id)
            summary_data['jira_status'] = jira_result
            
            # You could also run other tools here for a complete summary
            
            return {
                "status": "success",
                "app_id": app_id,
                "summary": summary_data,
                "generated_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error generating migration summary: {str(e)}")
            return {
                "status": "error",
                "message": str(e)
            }

# Example usage and demonstration
async def main():
    """Main function to demonstrate the Disha OCP Agent"""
    
    # Initialize the agent
    project_id = "your-google-cloud-project-id"  # Replace with actual project ID
    agent = DishaOCPAgent(project_id)
    
    # Optional: Initialize Jira tool with credentials
    # agent.initialize_jira_tool("https://your-jira.atlassian.net", "username", "api_token")
    
    # Example conversations
    test_scenarios = [
        {
            "message": "Can you assess the readiness of our Java Spring Boot microservices application for OCP migration?",
            "app_id": "app-123"
        },
        {
            "message": "I need to run security gate checks for our application",
            "app_id": "app-456"
        },
        {
            "message": "What are the data storage recommendations for our high-volume transactional application with ACID requirements?",
            "app_id": "app-789"
        },
        {
            "message": "Show me all Jira tickets for this application",
            "app_id": "app-123"
        },
        {
            "message": "What's the overall migration status and next steps?",
            "app_id": "app-123"
        }
    ]
    
    print("🚢 Disha OCP Migration Agent Initialized")
    print("=" * 60)
    print("Powered by Google ADK with FAISS similarity search")
    print("=" * 60)
    
    for i, scenario in enumerate(test_scenarios, 1):
        print(f"\n--- Scenario {i} ---")
        print(f"👤 User: {scenario['message']}")
        print(f"📱 App ID: {scenario['app_id']}")
        
        try:
            response = await agent.chat(scenario['message'], scenario['app_id'])
            print(f"🤖 Disha: {response}")
        except Exception as e:
            print(f"❌ Error: {str(e)}")
            
        print("-" * 60)
        
    # Example of getting migration summary
    print(f"\n--- Migration Summary Example ---")
    summary = await agent.get_migration_summary("app-123")
    print(f"📊 Summary: {json.dumps(summary, indent=2, default=str)}")

# Additional utility functions for the Disha OCP Agent

class MigrationWorkflowOrchestrator:
    """Orchestrates the complete migration workflow using all tools"""
    
    def __init__(self, agent: DishaOCPAgent):
        self.agent = agent
        
    async def run_complete_assessment(self, app_id: str, app_description: str) -> Dict[str, Any]:
        """Run complete migration assessment workflow"""
        workflow_results = {
            "app_id": app_id,
            "workflow_status": "running",
            "steps": {},
            "overall_readiness": None,
            "recommendations": [],
            "next_actions": []
        }
        
        try:
            # Step 1: Intake Assessment
            print(f"🔍 Running intake assessment for {app_id}...")
            intake_result = await self.agent.intake_tool.execute(app_description=app_description)
            workflow_results["steps"]["intake_assessment"] = intake_result
            
            if intake_result["status"] == "success":
                readiness_score = intake_result["assessment"]["readiness_score"]
                workflow_results["overall_readiness"] = readiness_score
                
                # Step 2: Run appropriate gate checks based on readiness
                if readiness_score >= 0.7:
                    print(f"✅ High readiness detected. Running comprehensive gate checks...")
                    gate_checks = ["security", "performance", "data", "network"]
                else:
                    print(f"⚠️ Medium/Low readiness. Running basic gate checks...")
                    gate_checks = ["security", "configuration"]
                
                gate_results = []
                for check_type in gate_checks:
                    gate_result = await self.agent.hardgate_tool.execute(
                        check_type=check_type, 
                        app_id=app_id
                    )
                    gate_results.append(gate_result)
                    
                workflow_results["steps"]["gate_checks"] = gate_results
                
                # Step 3: Data Analysis
                print(f"📊 Analyzing data requirements...")
                data_result = await self.agent.appdata_tool.execute(
                    app_description=app_description,
                    data_requirements="standard migration requirements"
                )
                workflow_results["steps"]["data_analysis"] = data_result
                
                # Step 4: Jira Status Check
                print(f"🎫 Checking Jira tickets...")
                jira_result = await self.agent.jira_tool.execute(app_id=app_id)
                workflow_results["steps"]["jira_status"] = jira_result
                
                # Generate overall recommendations
                workflow_results["recommendations"] = self._generate_workflow_recommendations(workflow_results)
                workflow_results["next_actions"] = self._generate_next_actions(workflow_results)
                workflow_results["workflow_status"] = "completed"
                
            else:
                workflow_results["workflow_status"] = "failed"
                workflow_results["recommendations"] = ["Fix intake assessment issues before proceeding"]
                
        except Exception as e:
            logger.error(f"Error in workflow orchestration: {str(e)}")
            workflow_results["workflow_status"] = "error"
            workflow_results["error"] = str(e)
            
        return workflow_results
        
    def _generate_workflow_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """Generate comprehensive recommendations based on all workflow results"""
        recommendations = []
        
        # Based on intake assessment
        if "intake_assessment" in results["steps"]:
            intake = results["steps"]["intake_assessment"]
            if intake["status"] == "success":
                readiness = intake["assessment"]["readiness_score"]
                if readiness >= 0.8:
                    recommendations.append("✅ Application is highly ready for OCP migration")
                elif readiness >= 0.6:
                    recommendations.append("⚠️ Application has moderate readiness - address identified issues")
                else:
                    recommendations.append("🚨 Application requires significant preparation before migration")
                    
        # Based on gate checks
        if "gate_checks" in results["steps"]:
            failed_gates = []
            for gate in results["steps"]["gate_checks"]:
                if gate.get("overall_status") == "FAILED":
                    failed_gates.extend([r["gate_type"] for r in gate.get("validation_results", []) if not r["passed"]])
                    
            if failed_gates:
                recommendations.append(f"🔧 Address failed gates: {', '.join(set(failed_gates))}")
            else:
                recommendations.append("✅ All gate validations passed")
                
        # Based on data analysis
        if "data_analysis" in results["steps"]:
            data = results["steps"]["data_analysis"]
            if data["status"] == "success":
                complexity = data["primary_recommendations"]["complexity_level"]
                if complexity == "High":
                    recommendations.append("📊 High data complexity - plan for extended migration window")
                elif complexity == "Medium":
                    recommendations.append("📊 Moderate data complexity - standard migration approach recommended")
                else:
                    recommendations.append("📊 Low data complexity - fast-track migration possible")
                    
        return recommendations
        
    def _generate_next_actions(self, results: Dict[str, Any]) -> List[str]:
        """Generate next action items based on workflow results"""
        actions = []
        
        # Check readiness score
        readiness = results.get("overall_readiness", 0)
        
        if readiness >= 0.8:
            actions.extend([
                "📋 Schedule migration planning meeting",
                "🔧 Prepare containerization scripts",
                "📅 Book migration window"
            ])
        elif readiness >= 0.6:
            actions.extend([
                "🔍 Address identified readiness gaps",
                "🧪 Conduct pilot migration test",
                "📋 Update migration timeline"
            ])
        else:
            actions.extend([
                "🚨 Complete application assessment",
                "🔧 Implement required architectural changes",
                "📚 Team training on OCP migration"
            ])
            
        # Add Jira-specific actions
        if "jira_status" in results["steps"]:
            jira = results["steps"]["jira_status"]
            if jira["status"] == "success":
                todo_count = jira["statistics"]["status_distribution"].get("To Do", 0)
                if todo_count > 0:
                    actions.append(f"🎫 Address {todo_count} pending Jira tickets")
                    
        return actions

# Enhanced CLI interface for the Disha OCP Agent
class DishaCLI:
    """Command-line interface for Disha OCP Agent"""
    
    def __init__(self, agent: DishaOCPAgent):
        self.agent = agent
        self.orchestrator = MigrationWorkflowOrchestrator(agent)
        
    async def interactive_mode(self):
        """Run interactive chat mode"""
        print("🚢 Welcome to Disha OCP Migration Assistant")
        print("Type 'help' for commands, 'quit' to exit")
        print("-" * 50)
        
        current_app_id = None
        
        while True:
            try:
                user_input = input(f"\n[{current_app_id or 'No App'}] 👤 You: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye! Happy migrating!")
                    break
                    
                elif user_input.lower() == 'help':
                    self._show_help()
                    continue
                    
                elif user_input.startswith('/app '):
                    current_app_id = user_input[5:].strip()
                    print(f"📱 Set current app ID to: {current_app_id}")
                    continue
                    
                elif user_input.startswith('/workflow '):
                    if not current_app_id:
                        print("❌ Please set an app ID first using '/app <app_id>'")
                        continue
                        
                    app_description = user_input[10:].strip()
                    print(f"🔄 Running complete migration workflow...")
                    
                    workflow_result = await self.orchestrator.run_complete_assessment(
                        current_app_id, app_description
                    )
                    
                    self._display_workflow_results(workflow_result)
                    continue
                    
                elif user_input.startswith('/summary'):
                    if not current_app_id:
                        print("❌ Please set an app ID first using '/app <app_id>'")
                        continue
                        
                    summary = await self.agent.get_migration_summary(current_app_id)
                    print(f"📊 Migration Summary:")
                    print(json.dumps(summary, indent=2, default=str))
                    continue
                    
                # Regular chat
                response = await self.agent.chat(user_input, current_app_id)
                print(f"🤖 Disha: {response}")
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye! Happy migrating!")
                break
            except Exception as e:
                print(f"❌ Error: {str(e)}")
                
    def _show_help(self):
        """Show help information"""
        print("""
        🚢 Disha OCP Migration Assistant - Commands:
        
        Chat Commands:
        - Just type your question or request
        - Ask about migration assessments, gate checks, data analysis, or Jira tickets
        
        Special Commands:
        /app <app_id>              - Set current application ID
        /workflow <app_description> - Run complete migration workflow
        /summary                   - Get migration summary for current app
        help                       - Show this help
        quit, exit, q             - Exit the application
        
        Example Questions:
        - "Assess my Java Spring Boot application"
        - "Run security gate checks"
        - "What storage do you recommend for my database?"
        - "Show me Jira tickets"
        """)
        
    def _display_workflow_results(self, results: Dict[str, Any]):
        """Display workflow results in a formatted way"""
        print(f"\n📋 Workflow Results for {results['app_id']}")
        print("=" * 50)
        
        if results["workflow_status"] == "completed":
            print(f"✅ Status: {results['workflow_status'].upper()}")
            
            if results["overall_readiness"]:
                readiness = results["overall_readiness"]
                status_emoji = "🟢" if readiness >= 0.8 else "🟡" if readiness >= 0.6 else "🔴"
                print(f"{status_emoji} Overall Readiness: {readiness:.2f}")
                
            print(f"\n💡 Recommendations:")
            for rec in results["recommendations"]:
                print(f"  • {rec}")
                
            print(f"\n📋 Next Actions:")
            for action in results["next_actions"]:
                print(f"  • {action}")
                
        else:
            print(f"❌ Status: {results['workflow_status'].upper()}")
            if "error" in results:
                print(f"Error: {results['error']}")

# Main execution with enhanced features
if __name__ == "__main__":
    import argparse
    
    async def run_disha():
        """Run Disha with command-line arguments"""
        parser = argparse.ArgumentParser(description="Disha OCP Migration Assistant")
        parser.add_argument("--project-id", required=True, help="Google Cloud Project ID")
        parser.add_argument("--location", default="us-central1", help="Google Cloud Location")
        parser.add_argument("--jira-url", help="Jira URL for integration")
        parser.add_argument("--jira-username", help="Jira username")
        parser.add_argument("--jira-token", help="Jira API token")
        parser.add_argument("--interactive", action="store_true", help="Run in interactive mode")
        parser.add_argument("--workflow", help="Run workflow for app (format: 'app_id:description')")
        
        args = parser.parse_args()
        
        # Initialize agent
        print("🚀 Initializing Disha OCP Agent...")
        agent = DishaOCPAgent(args.project_id, args.location)
        
        # Configure Jira if provided
        if args.jira_url and args.jira_username and args.jira_token:
            agent.initialize_jira_tool(args.jira_url, args.jira_username, args.jira_token)
            print("🔗 Jira integration configured")
            
        # Run in different modes
        if args.interactive:
            cli = DishaCLI(agent)
            await cli.interactive_mode()
            
        elif args.workflow:
            app_id, description = args.workflow.split(":", 1)
            orchestrator = MigrationWorkflowOrchestrator(agent)
            result = await orchestrator.run_complete_assessment(app_id.strip(), description.strip())
            print(json.dumps(result, indent=2, default=str))
            
        else:
            # Run demo scenarios
            await main()
    
    # Run the application
    asyncio.run(run_disha())