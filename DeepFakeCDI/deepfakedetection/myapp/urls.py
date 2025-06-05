from django.contrib import admin
from django.urls import path,include
from django.conf.urls.static import static
from django.conf import settings
from . import views
from django.urls import reverse


urlpatterns = [
    path('admin/', admin.site.urls),
    path('',views.landing,name='landing'),
    path('dashboard/',views.dashboard,name='dashboard'),
    # path("upload-video/", views.upload_video, name="upload_video"),
    path('process-video/', views.process_video, name='process_video'),
    # path('get-pro/',views.get_pro,name='get-pro'),
    path('pricing/',views.pricing,name='pricing'),
    path('register/', views.register_view, name='register'),
    path('login/', views.login_view, name='login'),
    path('logout/', views.logout_view, name='logout'),
    path('temp_dashboard/', views.temp_dashboard, name='temp_dashboard'),
    path('blogs/',views.blogs,name='blogs'),
    path('api/contact/', views.process_contact_form, name='process_contact_form'),
    path('test-email/', views.test_email, name='test_email'),
    path('use-case/',views.use_case,name='use_case'),
              ]+ static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
