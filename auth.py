import os
from datetime import datetime

import streamlit as st
from google.auth.transport.requests import Request
from google.oauth2 import id_token
from google_auth_oauthlib.flow import Flow
from sqlalchemy.orm import Session

from database import User, get_db

# Google OAuth Configuration
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")


def get_google_oauth_flow():
    """Create Google OAuth flow"""
    return Flow.from_client_config(
        client_config={
            "web": {
                "client_id": GOOGLE_CLIENT_ID,
                "client_secret": GOOGLE_CLIENT_SECRET,
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token",
                "redirect_uris": ["http://localhost:8501/callback"],
            }
        },
        scopes=[
            "openid",
            "https://www.googleapis.com/auth/userinfo.email",
            "https://www.googleapis.com/auth/userinfo.profile",
        ],
    )


def authenticate_google_user(db: Session, token):
    """
    Authenticate user with Google token and store/update user in database
    """
    try:
        # Verify Google token and get user info
        id_info = id_token.verify_oauth2_token(token, Request(), GOOGLE_CLIENT_ID)

        # Extract user details
        google_id = id_info["sub"]
        email = id_info["email"]
        name = id_info.get("name", "")
        picture_url = id_info.get("picture", "")

        # Check if user exists
        user = db.query(User).filter(User.google_id == google_id).first()

        if not user:
            # Create new user
            user = User(
                google_id=google_id,
                email=email,
                name=name,
                picture_url=picture_url,
                last_login=datetime.utcnow(),
            )
            db.add(user)
        else:
            # Update last login
            user.last_login = datetime.utcnow()

        db.commit()
        db.refresh(user)

        return user

    except ValueError:
        return None


def login_flow():
    """
    Streamlit login flow with Google OAuth
    """
    # Initialize database session
    db = next(get_db())

    # Check if user is already logged in
    if "user" in st.session_state and st.session_state.user:
        return st.session_state.user

    # Create Google OAuth flow
    flow = get_google_oauth_flow()

    # Authorization URL
    authorization_url, _ = flow.authorization_url(
        access_type="offline", prompt="consent"
    )

    # Login button
    if st.button("Login with Google"):
        st.markdown(
            f'<a href="{authorization_url}" target="_self">Login</a>',
            unsafe_allow_html=True,
        )

    # Handle callback and token exchange
    auth_code = st.query_params.get("code", [None])[0]
    if auth_code:
        try:
            # Exchange auth code for token
            flow.fetch_token(code=auth_code)
            credentials = flow.credentials

            # Authenticate user
            user = authenticate_google_user(db, credentials.id_token)

            if user:
                st.session_state.user = user
                st.success(f"Logged in as {user.name}")
                return user
            else:
                st.error("Authentication failed")
        except Exception as e:
            st.error(f"Login error: {e}")

    return None

