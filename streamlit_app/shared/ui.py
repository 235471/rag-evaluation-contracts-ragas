"""
Shared UI components for Streamlit apps.
"""

import streamlit as st


def inject_custom_css():
    """Inject custom CSS for premium design."""
    st.markdown(
        """
        <style>
        /* Custom scrollbar */
        ::-webkit-scrollbar {
            width: 8px;
            height: 8px;
        }
        ::-webkit-scrollbar-track {
            background: #1e293b;
        }
        ::-webkit-scrollbar-thumb {
            background: #4F46E5;
            border-radius: 4px;
        }
        
        /* Card-like containers */
        .stExpander {
            border: 1px solid #334155;
            border-radius: 8px;
        }
        
        /* Smooth transitions */
        .stButton > button {
            transition: all 0.2s ease;
        }
        .stButton > button:hover {
            transform: translateY(-1px);
            box-shadow: 0 4px 12px rgba(79, 70, 229, 0.3);
        }
        
        /* Chat messages */
        .stChatMessage {
            border-radius: 12px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar_header(title: str = "⚙️ Configurações"):
    """Render common sidebar header."""
    st.header(title)
    st.divider()


def render_footer():
    """Render common footer."""
    st.divider()
    st.caption("LangChain Advanced RAG MVP • Powered by Streamlit")
