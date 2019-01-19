from django import template
import datetime
register = template.Library()

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
