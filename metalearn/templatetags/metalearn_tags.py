# -*- coding: utf-8 -*-
import os.path

from cruds_adminlte import utils
from django import template

from django.urls import (reverse, NoReverseMatch)  # django2.0
from django.db import models
from django.utils import six
from django.utils.html import escape
from django.utils.safestring import mark_safe
from django import template
import datetime



register = template.Library()

register = template.Library()
if hasattr(register, 'assignment_tag'):
    register_tag = register.assignment_tag
else:
    register_tag = register.simple_tag


@register.filter
def format_uS2time(value):
    if value == "":
        return ""
    return datetime.timedelta(seconds=float(value)/1000000).__str__().split(".")[0]

@register.filter
def format_uS2ms(value):    
    if value == "":
        return ""
    return "%s" % round(value/1000.0,2)

    
@register.filter
def format_number(value):
    if value == "":
        return ""
    r =  "{:,}".format(int(value))
    return r.replace(",",".")

@register.filter
def format_float(value):
    if value == "":
        return ""
    r =  "{:,}".format(round(float(value),6))
    return r.replace(",",".")


@register_tag
def crud_url(obj, action, namespace=None):
    try:
        nurl = utils.crud_url_name(type(obj), action)
        if namespace:
            nurl = namespace + ':' + nurl
        if action in utils.LIST_ACTIONS:
            url = reverse(nurl)
        else:
            url = reverse(nurl, kwargs={'pk': obj.pk})
    except NoReverseMatch:
        url = None
    return url


@register.filter
def format_value_fk(obj, field_name):
    """
    Simple value formatting.

    If value is model instance returns link to detail view if exists.
    """
    if '__' in field_name:
        related_model, field_name = field_name.split('__', 1)
        obj = getattr(obj, related_model)
    display_func = getattr(obj, 'get_%s_display' % field_name, None)
    if display_func:
        return display_func()
    value = getattr(obj, field_name)

    if isinstance(value, models.Model):
        url = crud_url(value, utils.ACTION_DETAIL)
        if url:
            return mark_safe('<a href="%s">%s</a>' % (url, escape(value)))
        else:
            if hasattr(value, 'get_absolute_url'):
                url = getattr(value, 'get_absolute_url')()
                return mark_safe('<a href="%s">%s</a>' % (url, escape(value)))
    if value is None:
        value = ""
    return value