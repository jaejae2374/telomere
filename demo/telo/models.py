from django.db import models
from datetime import datetime
# Create your models here.

def image_name(instance, filename):
    ext = filename.split('.')[-1]
    return datetime.now().strftime(f"%Y-%m-%dT%H-%M-%S.{ext}")

class Scalp(models.Model):
    
    id = models.AutoField(primary_key=True)
    name = models.CharField(max_length=100, blank=True)
    image = models.ImageField(upload_to=image_name, null=True)
