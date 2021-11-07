from django.shortcuts import render, redirect
from django.views.decorators.csrf import csrf_protect
from django.contrib.auth import *
from django.contrib.auth.forms import UserCreationForm
from django.http import HttpResponse


# Create your views here.
@csrf_protect
def user_login(request):
    if request.POST:
        username = request.POST.get('username')
        password = request.POST.get('password')
        user = authenticate(username=username, password=password)
        if user is not None and user.is_active:
            print("User Login:  Username:" + username + '    Password:' + password)
            print(request.POST)
            login(request, user)
            return redirect(request.POST.get('path'))
        else:
            return render(request, 'base.html', {'message': 'Username or Password wrong!'})
    else:
        return render(request, '404.html')


def user_logout(request):
    logout(request)
    return HttpResponse()


@csrf_protect
def user_register(request):
    if request.method == 'POST':
        form = UserCreationForm(request.POST)
        if form.is_valid():
            new_user = form.save()
            return render(request, 'base.html', {'message': 'Registered successfully, congratulations! Please login.'})
        else:
            return render(request, 'register.html', {'error': 'Invalid input!', 'form': UserCreationForm()})
    else:
        return render(request, "register.html", {'form': UserCreationForm()})

