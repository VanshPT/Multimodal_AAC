from django.urls import path
from django.views.generic import RedirectView

from . import views
from .aac import views as aac_views

urlpatterns = [
    path('', views.landing, name="landing"),
    path('dashboard/<str:username>/', views.render_dashboard, name="render_dashboard"),
    path('auto_clustering/<str:username>/', views.auto_clustering, name="auto_clustering"),
    path('explore_topics/<str:username>/', views.explore_topics, name="explore_topics"),
    path('search_papers/<str:username>/', views.search_papers, name="search_papers"),
    path("aac", RedirectView.as_view(url="/aac/", permanent=False), name="aac_session_slash_redirect"),
    path("aac/", aac_views.aac_session_page, name="aac_session"),
    path("aac/api/users/", aac_views.list_users_api, name="aac_users_api"),
    path("aac/api/start_session/", aac_views.start_session_api, name="aac_start_session_api"),
    path("aac/api/partner_message/", aac_views.partner_message_api, name="aac_partner_message_api"),
    path("aac/api/speak_mode/", aac_views.speak_mode_api, name="aac_speak_mode_api"),
    path("aac/api/confirm_response/", aac_views.confirm_response_api, name="aac_confirm_response_api"),
    path("aac/api/session_state/", aac_views.session_state_api, name="aac_session_state_api"),
    path("aac/api/livekit/listen_token/", aac_views.livekit_listen_token_api, name="aac_livekit_listen_token_api"),
    path("aac/api/livekit/transcript/", aac_views.livekit_transcript_api, name="aac_livekit_transcript_api"),
    path("aac/api/livekit/ingest_transcript/", aac_views.livekit_ingest_transcript_api, name="aac_livekit_ingest_transcript_api"),
    path("aac/api/livekit/tts_token/", aac_views.livekit_tts_token_api, name="aac_livekit_tts_token_api"),
    path("health/llm", aac_views.llm_health_api, name="aac_llm_health_api"),
    path("metrics/", aac_views.metrics_page, name="aac_metrics_page"),
    path("metrics/download/", aac_views.download_metrics_logs, name="aac_metrics_download"),
]
