from django.contrib import admin
from django.urls import path,include
from django.conf.urls.static import static
from django.conf import settings
from . import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('',views.landing,name='landing'),
    path('dashboard/',views.dashboard,name='dashboard'),
    path("upload-video/", views.upload_video, name="upload_video"),
    path('process-video/', views.process_video, name='process_video'),

]+ static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
