# Generated by Django 5.1.3 on 2024-11-17 01:55

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('scanner', '0002_rename_image_furnituredetection_image_data_and_more'),
    ]

    operations = [
        migrations.AddField(
            model_name='room',
            name='height',
            field=models.FloatField(default=0.0),
        ),
        migrations.AddField(
            model_name='room',
            name='length',
            field=models.FloatField(default=0.0),
        ),
        migrations.AddField(
            model_name='room',
            name='width',
            field=models.FloatField(default=0.0),
        ),
    ]
