import os
import base64
import time
import re
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer
from PIL import Image
import PyPDF2
import docx
import random
import requests
from groq import Groq


# ========== GROK API CONFIGURATION ==========
# Set your Grok API key in Streamlit secrets or environment variable
# For Streamlit Cloud: Add to secrets.toml
# GROK_API_KEY = "your-key-here"
GROQ_MODEL = "llama-3.1-8b-instant"
# ============================================

# ========== PRE-LOAD CONFIGURATION ==========
PRELOAD_TEXT_DIR = "preload_data/fact_checks"
PRELOAD_IMAGE_DIR = "preload_data/evidence"
PRELOAD_AUDIO_DIR = "preload_data/audio_claims"
# ===========================================

# Audio support
try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False


def get_groq_client():
    """Initialize Groq API client"""
    api_key = os.environ.get("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY")
    if not api_key:
        st.error("⚠️ GROQ_API_KEY not found. Please set it in secrets or environment variables.")
        st.stop()
    return Groq(api_key=api_key)


def analyze_with_groq(prompt: str, model: str = GROQ_MODEL) -> str:
    """Generate response using Groq API"""
    try:
        client = get_groq_client()
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Groq API Error: {e}")
        return f"Error generating response: {str(e)}"



def analyze_image_with_groq(image_path: str, prompt: str = None) -> str:
    """Analyze image (fallback mode — Groq doesn't support vision yet)"""
    st.warning("⚠️ Groq API currently does not support image analysis. Skipping visual processing.")
    return f"Image: {os.path.basename(image_path)} (visual analysis not available)"



class MisinformationDetectionRAG:
    def __init__(self, db_path: str = "./misinfo_rag_db"):
        """Initialize the misinformation detection RAG system"""
        self.db_path = db_path
        self.client = chromadb.PersistentClient(path=db_path)
        
        # Cache embedding model
        if 'embedding_model' not in st.session_state:
            with st.spinner("Loading embedding model..."):
                st.session_state.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.embedding_model = st.session_state.embedding_model
        
        # Cache Whisper model
        if WHISPER_AVAILABLE and 'whisper_model' not in st.session_state:
            with st.spinner("Loading Whisper model..."):
                st.session_state.whisper_model = whisper.load_model("base")
        
        if WHISPER_AVAILABLE:
            self.whisper_model = st.session_state.whisper_model
        
        # Collections for different content types
        self.fact_checks = self.client.get_or_create_collection(
            name="fact_checks",
            metadata={"description": "Verified fact-checks and debunked claims"}
        )
        
        self.evidence = self.client.get_or_create_collection(
            name="evidence",
            metadata={"description": "Evidence base including documents and media"}
        )
    
    def _embed(self, text: str) -> List[float]:
        """Generate embedding for semantic search"""
        return self.embedding_model.encode(text).tolist()
    
    def _transcribe_audio(self, audio_path: str) -> Dict[str, Any]:
        """Transcribe audio with timestamps using Whisper"""
        if not WHISPER_AVAILABLE:
            return {
                'text': f"Audio: {os.path.basename(audio_path)}",
                'segments': []
            }
        
        try:
            result = self.whisper_model.transcribe(audio_path, verbose=False)
            return {
                'text': result["text"],
                'segments': result.get("segments", [])
            }
        except Exception as e:
            st.error(f"Error transcribing audio: {e}")
            return {
                'text': f"Audio: {os.path.basename(audio_path)}",
                'segments': []
            }
    
    def _extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF"""
        try:
            text = ""
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
            return text.strip()
        except Exception as e:
            st.error(f"Error extracting PDF: {e}")
            return f"PDF: {os.path.basename(pdf_path)}"
    
    def add_fact_check(self, claim: str, verdict: str, explanation: str, 
                       source: str = None, metadata: Dict = None) -> str:
        """Add a verified fact-check to the database"""
        doc_id = f"factcheck_{int(time.time() * 1000)}_{self.fact_checks.count()}"
        
        content = f"""CLAIM: {claim}
VERDICT: {verdict}
EXPLANATION: {explanation}
SOURCE: {source or 'User submission'}"""
        
        meta = metadata or {}
        meta.update({
            'type': 'fact_check',
            'claim': claim,
            'verdict': verdict,
            'source': source or 'user',
            'added_at': datetime.now().isoformat(),
        })
        
        self.fact_checks.add(
            ids=[doc_id],
            embeddings=[self._embed(claim)],
            documents=[content],
            metadatas=[meta]
        )
        return doc_id
    
    def add_evidence(self, content: str, content_type: str, 
                     source: str = None, metadata: Dict = None) -> str:
        """Add evidence document (text, image description, audio transcript)"""
        doc_id = f"evidence_{content_type}_{int(time.time() * 1000)}_{self.evidence.count()}"
        
        meta = metadata or {}
        meta.update({
            'type': content_type,
            'source': source or 'user',
            'added_at': datetime.now().isoformat(),
            'preview': content[:200]
        })
        
        self.evidence.add(
            ids=[doc_id],
            embeddings=[self._embed(content)],
            documents=[content],
            metadatas=[meta]
        )
        return doc_id
    
    def add_image_evidence(self, path: str, metadata: Dict = None) -> Tuple[str, str]:
        """Add image evidence with Grok Vision analysis"""
        analysis = analysis = analyze_image_with_groq(path)

        
        meta = metadata or {}
        meta.update({
            'path': path,
            'filename': os.path.basename(path)
        })
        
        doc_id = self.add_evidence(analysis, 'image', metadata=meta)
        return doc_id, analysis
    
    def add_audio_evidence(self, path: str, metadata: Dict = None) -> Tuple[str, str]:
        """Add audio evidence with transcription"""
        transcription_data = self._transcribe_audio(path)
        transcription = transcription_data['text']
        
        meta = metadata or {}
        meta.update({
            'path': path,
            'filename': os.path.basename(path)
        })
        
        doc_id = self.add_evidence(transcription, 'audio', metadata=meta)
        return doc_id, transcription
    
    def add_document_evidence(self, path: str, metadata: Dict = None) -> Tuple[str, str]:
        """Add document evidence"""
        if path.endswith('.pdf'):
            content = self._extract_text_from_pdf(path)
        elif path.endswith('.docx'):
            doc = docx.Document(path)
            content = "\n".join([p.text for p in doc.paragraphs])
        else:
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
        
        meta = metadata or {}
        meta.update({
            'path': path,
            'filename': os.path.basename(path)
        })
        
        doc_id = self.add_evidence(content, 'document', metadata=meta)
        return doc_id, content
    
    def verify_claim(self, claim: str, n_results: int = 5) -> Dict[str, Any]:
        """Verify a claim against fact-check database and evidence"""
        # Search fact-checks
        fact_check_results = []
        if self.fact_checks.count() > 0:
            fc_results = self.fact_checks.query(
                query_embeddings=[self._embed(claim)],
                n_results=min(n_results, self.fact_checks.count())
            )
            fact_check_results = self._format_results(fc_results)
        
        # Search evidence
        evidence_results = []
        if self.evidence.count() > 0:
            ev_results = self.evidence.query(
                query_embeddings=[self._embed(claim)],
                n_results=min(n_results, self.evidence.count())
            )
            evidence_results = self._format_results(ev_results)
        
        # Generate verification using Grok
        context = self._build_verification_context(claim, fact_check_results, evidence_results)
        
        prompt = f"""You are a fact-checking assistant. Analyze this claim and provide a detailed verification report.

CLAIM TO VERIFY: {claim}

{context}

Provide a structured analysis:
1. **VERDICT**: [TRUE / FALSE / PARTIALLY TRUE / UNVERIFIED / MISLEADING]
2. **CONFIDENCE**: [0-100%]
3. **KEY FINDINGS**: Bullet points of main facts
4. **EVIDENCE ASSESSMENT**: Quality and reliability of available evidence
5. **CONTEXT**: Important context or nuance
6. **RECOMMENDATION**: What the public should know
7. **RED FLAGS**: Any concerning patterns or manipulation indicators

Be objective, cite evidence, and explain your reasoning clearly."""
        
        analysis =analyze_with_groq(prompt)
        
        return {
            'claim': claim,
            'analysis': analysis,
            'fact_checks': fact_check_results,
            'evidence': evidence_results,
            'timestamp': datetime.now().isoformat()
        }
    
    def _format_results(self, results: Dict) -> List[Dict]:
        """Format ChromaDB results"""
        formatted = []
        if results['ids'][0]:
            for i in range(len(results['ids'][0])):
                formatted.append({
                    'id': results['ids'][0][i],
                    'content': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'relevance': 1 / (1 + results['distances'][0][i])
                })
        return formatted
    
    def _build_verification_context(self, claim: str, fact_checks: List[Dict], 
                                   evidence: List[Dict]) -> str:
        """Build context for verification"""
        context_parts = []
        
        if fact_checks:
            context_parts.append("RELATED FACT-CHECKS:")
            for i, fc in enumerate(fact_checks[:3], 1):
                context_parts.append(f"\n[Fact-Check {i}]\n{fc['content']}\n")
        
        if evidence:
            context_parts.append("\nRELATED EVIDENCE:")
            for i, ev in enumerate(evidence[:3], 1):
                context_parts.append(f"\n[Evidence {i}]\n{ev['content'][:500]}...\n")
        
        if not context_parts:
            return "NO PRIOR FACT-CHECKS OR EVIDENCE AVAILABLE. Analyze based on general knowledge."
        
        return "\n".join(context_parts)
    
    def monitor_claim(self, claim: str, keywords: List[str] = None) -> Dict[str, Any]:
        """Monitor for emerging misinformation patterns"""
        # This would integrate with real-time data sources in production
        # For now, simulate monitoring against database
        
        results = self.verify_claim(claim, n_results=10)
        
        # Pattern analysis
        pattern_prompt = f"""Analyze this claim for misinformation patterns:

CLAIM: {claim}
KEYWORDS: {', '.join(keywords) if keywords else 'None specified'}

Identify:
1. Similar debunked claims in history
2. Common manipulation tactics used
3. Likely spread vectors (social media, messaging apps)
4. Target audience and potential impact
5. Urgency level (LOW / MEDIUM / HIGH / CRITICAL)

Provide actionable monitoring recommendations."""
        
        pattern_analysis = analyze_with_groq(pattern_prompt)
        
        return {
            **results,
            'pattern_analysis': pattern_analysis,
            'keywords': keywords or []
        }
    
    def get_stats(self) -> Dict:
        """Get database statistics"""
        return {
            'fact_checks': self.fact_checks.count(),
            'evidence': self.evidence.count(),
            'total': self.fact_checks.count() + self.evidence.count()
        }
    
    def generate_public_update(self, claim: str, verification: Dict) -> str:
        """Generate easy-to-understand public update"""
        prompt = f"""Create a clear, concise public communication about this claim verification.

CLAIM: {claim}
VERIFICATION RESULT: {verification['analysis'][:500]}

Write in plain language for general public:
- Keep it brief (2-3 paragraphs)
- Use simple terms
- Highlight key facts
- Include actionable advice
- Maintain neutral, trustworthy tone

Format as a public announcement."""
        
        return analyze_with_groq(prompt)


def preload_fact_checks(rag: MisinformationDetectionRAG):
    """Preload sample fact-checks"""
    sample_checks = [
        {
            'claim': "Drinking hot water cures COVID-19",
            'verdict': "FALSE",
            'explanation': "No scientific evidence supports this. COVID-19 requires medical treatment.",
            'source': "WHO Fact-Check Database"
        },
        {
            'claim': "5G networks cause coronavirus spread",
            'verdict': "FALSE",
            'explanation': "Viruses cannot travel on radio waves. This has been thoroughly debunked.",
            'source': "CDC Misinformation Tracker"
        },
        {
            'claim': "Vaccines contain microchips for tracking",
            'verdict': "FALSE",
            'explanation': "Vaccines contain biological components only. No tracking devices exist.",
            'source': "FDA Official Statement"
        }
    ]
    
    for check in sample_checks:
        rag.add_fact_check(
            claim=check['claim'],
            verdict=check['verdict'],
            explanation=check['explanation'],
            source=check['source']
        )


def setup_page():
    """Setup Streamlit page"""
    st.set_page_config(
        page_title="Misinformation Detection System",
        page_icon="🛡️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.markdown("""
        <style>
        .main {padding: 1rem;}
        .verdict-true {color: #2ecc71; font-weight: bold;}
        .verdict-false {color: #e74c3c; font-weight: bold;}
        .verdict-partial {color: #f39c12; font-weight: bold;}
        .evidence-card {
            padding: 1rem;
            border: 1px solid #ddd;
            border-radius: 8px;
            margin: 0.5rem 0;
            background-color: #f9f9f9;
        }
        </style>
    """, unsafe_allow_html=True)


def init_session_state():
    """Initialize session state"""
    if 'rag' not in st.session_state:
        st.session_state.rag = MisinformationDetectionRAG()
        
        if st.session_state.rag.get_stats()['total'] == 0:
            with st.spinner("Loading initial fact-check database..."):
                preload_fact_checks(st.session_state.rag)
    
    if 'verification_history' not in st.session_state:
        st.session_state.verification_history = []


def sidebar():
    """Sidebar configuration"""
    with st.sidebar:
        st.title("🛡️ Misinfo Detection")
        st.caption("AI-Powered Fact-Checking System")
        
        st.markdown("---")
        
        # Statistics
        st.subheader("📊 Database Stats")
        stats = st.session_state.rag.get_stats()
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Fact-Checks", stats['fact_checks'])
        with col2:
            st.metric("Evidence", stats['evidence'])
        
        st.markdown("---")
        
        # Add fact-check
        st.subheader("➕ Add Fact-Check")
        with st.form("add_factcheck"):
            claim = st.text_area("Claim", height=100)
            verdict = st.selectbox("Verdict", ["TRUE", "FALSE", "PARTIALLY TRUE", "UNVERIFIED", "MISLEADING"])
            explanation = st.text_area("Explanation", height=100)
            source = st.text_input("Source")
            
            if st.form_submit_button("Add Fact-Check"):
                if claim and explanation:
                    st.session_state.rag.add_fact_check(claim, verdict, explanation, source)
                    st.success("✅ Fact-check added!")
                    st.rerun()
        
        st.markdown("---")
        
        # Add evidence
        st.subheader("📤 Add Evidence")
        evidence_type = st.radio("Evidence Type", ["Document", "Image", "Audio"])
        
        if evidence_type == "Document":
            uploaded = st.file_uploader("Upload Document", type=['txt', 'pdf', 'docx'])
            if uploaded and st.button("Add Document"):
                path = Path("data/evidence") / uploaded.name
                path.parent.mkdir(parents=True, exist_ok=True)
                with open(path, "wb") as f:
                    f.write(uploaded.getbuffer())
                with st.spinner("Processing..."):
                    st.session_state.rag.add_document_evidence(str(path))
                st.success("✅ Evidence added!")
                st.rerun()
        
        elif evidence_type == "Image":
            uploaded = st.file_uploader("Upload Image", type=['jpg', 'png', 'jpeg'])
            if uploaded and st.button("Add Image"):
                path = Path("data/evidence") / uploaded.name
                path.parent.mkdir(parents=True, exist_ok=True)
                with open(path, "wb") as f:
                    f.write(uploaded.getbuffer())
                with st.spinner("Analyzing image..."):
                    st.session_state.rag.add_image_evidence(str(path))
                st.success("✅ Image evidence added!")
                st.rerun()


def main_interface():
    """Main verification interface"""
    st.title("🔍 Claim Verification System")
    
    tab1, tab2, tab3 = st.tabs(["🔎 Verify Claim", "📡 Monitor", "📋 History"])
    
    with tab1:
        st.markdown("### Enter a claim to verify")
        claim_input = st.text_area(
            "Claim to verify",
            height=100,
            placeholder="Enter a claim, statement, or rumor you want to fact-check..."
        )
        
        col1, col2 = st.columns([1, 4])
        with col1:
            verify_btn = st.button("🔍 Verify", type="primary", use_container_width=True)
        
        if verify_btn and claim_input:
            with st.spinner("🔍 Verifying claim against database and evidence..."):
                result = st.session_state.rag.verify_claim(claim_input)
                st.session_state.verification_history.insert(0, result)
            
            # Display results
            st.markdown("---")
            st.markdown("### 📊 Verification Results")
            
            # Analysis
            st.markdown("#### 🤖 AI Analysis")
            st.markdown(result['analysis'])
            
            # Related fact-checks
            if result['fact_checks']:
                st.markdown("#### ✅ Related Fact-Checks")
                for fc in result['fact_checks'][:3]:
                    with st.expander(f"🔖 {fc['metadata'].get('claim', 'Fact-Check')[:80]}..."):
                        st.markdown(fc['content'])
                        st.caption(f"Relevance: {fc['relevance']:.0%}")
            
            # Evidence
            if result['evidence']:
                st.markdown("#### 📚 Supporting Evidence")
                for ev in result['evidence'][:3]:
                    with st.expander(f"📄 {ev['metadata'].get('filename', 'Evidence')[:60]}..."):
                        st.text(ev['content'][:500])
                        st.caption(f"Relevance: {ev['relevance']:.0%}")
            
            # Public update
            st.markdown("#### 📢 Public Communication")
            with st.spinner("Generating public-friendly summary..."):
                public_update = st.session_state.rag.generate_public_update(claim_input, result)
                st.info(public_update)
    
    with tab2:
        st.markdown("### 📡 Claim Monitoring")
        st.caption("Monitor emerging claims and detect patterns")
        
        monitor_claim = st.text_area("Claim to monitor", height=100)
        keywords = st.text_input("Keywords (comma-separated)", placeholder="covid, vaccine, 5g")
        
        if st.button("📡 Start Monitoring", type="primary"):
            keyword_list = [k.strip() for k in keywords.split(",")] if keywords else []
            with st.spinner("Analyzing patterns and threats..."):
                monitor_result = st.session_state.rag.monitor_claim(monitor_claim, keyword_list)
            
            st.markdown("#### 🔍 Verification Analysis")
            st.markdown(monitor_result['analysis'])
            
            st.markdown("#### 🎯 Pattern Analysis")
            st.markdown(monitor_result['pattern_analysis'])
    
    with tab3:
        st.markdown("### 📋 Verification History")
        if st.session_state.verification_history:
            for i, item in enumerate(st.session_state.verification_history[:10]):
                with st.expander(f"🔖 {item['claim'][:80]}... - {item['timestamp'][:19]}"):
                    st.markdown(item['analysis'])
        else:
            st.info("No verification history yet. Start by verifying a claim!")


def main():
    """Main application"""
    setup_page()
    init_session_state()
    sidebar()
    main_interface()
    
    # Footer
    st.markdown("---")
    st.caption("🛡️ Misinformation Detection System")


if __name__ == "__main__":
    main()
   
   
   
