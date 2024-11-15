# scanner/forms.py
from django import forms
from .models import Room
import json

class RoomForm(forms.ModelForm):
    """Form for creating and editing rooms."""
    length = forms.FloatField(
        min_value=0,
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
            'placeholder': 'Length in inches'
        })
    )
    width = forms.FloatField(
        min_value=0,
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
            'placeholder': 'Width in inches'
        })
    )
    height = forms.FloatField(
        min_value=0,
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
            'placeholder': 'Height in inches'
        })
    )

    class Meta:
        model = Room
        fields = ['name', 'description']
        widgets = {
            'name': forms.TextInput(attrs={
                'class': 'form-control',
                'placeholder': 'Enter room name'
            }),
            'description': forms.Textarea(attrs={
                'class': 'form-control',
                'placeholder': 'Enter room description',
                'rows': 3
            })
        }

    def __init__(self, *args, **kwargs):
        instance = kwargs.get('instance')
        if instance and instance.dimensions:
            # If we have an instance with dimensions, set initial values
            initial = kwargs.get('initial', {})
            initial.update({
                'length': instance.dimensions.get('length', 0),
                'width': instance.dimensions.get('width', 0),
                'height': instance.dimensions.get('height', 0)
            })
            kwargs['initial'] = initial
        super().__init__(*args, **kwargs)

    def clean(self):
        cleaned_data = super().clean()
        # Create dimensions JSON from the individual fields
        dimensions = {
            'length': cleaned_data.get('length', 0),
            'width': cleaned_data.get('width', 0),
            'height': cleaned_data.get('height', 0)
        }
        # Add dimensions to cleaned data
        cleaned_data['dimensions'] = dimensions
        return cleaned_data

    def save(self, commit=True):
        instance = super().save(commit=False)
        instance.dimensions = self.cleaned_data['dimensions']
        if commit:
            instance.save()
        return instance