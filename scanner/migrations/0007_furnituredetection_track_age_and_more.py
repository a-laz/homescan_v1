# Generated by Django 5.1.3 on 2024-11-28 21:09

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('scanner', '0006_scansession'),
    ]

    operations = [
        migrations.AddField(
            model_name='furnituredetection',
            name='track_age',
            field=models.IntegerField(default=0),
        ),
        migrations.AddField(
            model_name='furnituredetection',
            name='track_hits',
            field=models.IntegerField(default=1),
        ),
        migrations.AddField(
            model_name='furnituredetection',
            name='track_id',
            field=models.IntegerField(blank=True, null=True),
        ),
    ]
