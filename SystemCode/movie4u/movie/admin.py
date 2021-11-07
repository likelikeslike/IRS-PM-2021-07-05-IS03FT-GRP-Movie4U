from django.contrib import admin
from .models import *


class MovieAdmin(admin.ModelAdmin):
    list_display = ('title', 'movieid', 'rate')


admin.site.register(Movie, MovieAdmin)
