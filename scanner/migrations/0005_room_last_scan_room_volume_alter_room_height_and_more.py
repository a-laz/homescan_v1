# Generated by Django 5.1.3 on 2024-11-17 03:55

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('scanner', '0004_furnituredetection_frame_number_and_more'),
    ]

    operations = [
        migrations.AddField(
            model_name='room',
            name='last_scan',
            field=models.DateTimeField(auto_now=True),
        ),
        migrations.AddField(
            model_name='room',
            name='volume',
            field=models.FloatField(blank=True, null=True),
        ),
        migrations.AlterField(
            model_name='room',
            name='height',
            field=models.FloatField(blank=True, null=True),
        ),
        migrations.AlterField(
            model_name='room',
            name='length',
            field=models.FloatField(blank=True, null=True),
        ),
        migrations.AlterField(
            model_name='room',
            name='width',
            field=models.FloatField(blank=True, null=True),
        ),
    ]