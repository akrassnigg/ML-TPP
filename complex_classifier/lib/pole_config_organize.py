# -*- coding: utf-8 -*-
"""
Created on Tue Aug 10 11:09:38 2021

@author: siegfriedkaidisch

Functions, that organize pole configurations

"""


def pole_config_organize_re2(pole_class, pole_params):
    '''
    Organize pole configs such that:
        
        Parameters of real poles are always kept at the front
        
        Poles are ordered by Re(pole position)**2 from small to large
        
        Im(Pole-position)>0 (convention, see parameters file)
        
    Note: Assumes poles to be ordered like in get_train_params(), but with imaginary parts of real poles removed (see pole_curve_calc vs pole_curve_calc2)
        
    pole_class: int = 0-8 
        The class of the pole configuration to be found
        
    pole_params: numpy.ndarray or torch.Tensor of shape (m,k), where m is the number of samples and k depends on the pole class (e.g. k=2 for pole_class=0)
        Pole configurations to be organized
        
    returns: numpy.ndarray or torch.Tensor of shape (m,k)
        The organized pole configurations
    '''
    if pole_class == 0:
        None
        
    elif pole_class == 1:
        #make sure that Im(Pole-position)>0):
        indices = pole_params[:,1]    <  0
        pole_params[indices,1]       *= -1
        pole_params[indices,3]       *= -1
        
    elif pole_class == 2:
        #Order poles by Re(Pole-position)**2
        params1 = pole_params[:, 0:2]
        params2 = pole_params[:, 2:4]
        val1    = params1[:,0]**2
        val2    = params2[:,0]**2
        swap    = ~(val1 < val2)
        pole_params[swap,:] = pole_params[swap][:, [2,3,  0,1]]
        
    elif pole_class == 3:
        #make sure that Im(Pole-position)>0):
        indices = pole_params[:,3]    <  0
        pole_params[indices,3]       *= -1
        pole_params[indices,5]       *= -1
        
    elif pole_class == 4:
        #make sure that Im(Pole-position)>0):
        indices = pole_params[:,1]    <  0
        pole_params[indices,1]       *= -1
        pole_params[indices,3]       *= -1
        #make sure that Im(Pole-position)>0):
        indices = pole_params[:,5]    <  0
        pole_params[indices,5]       *= -1
        pole_params[indices,7]       *= -1
        
        #Order poles by Re(Pole-position)**2
        params1 = pole_params[:, 0:4]
        params2 = pole_params[:, 4:8]
        val1    = params1[:,0]**2
        val2    = params2[:,0]**2
        swap    = ~(val1 < val2)
        pole_params[swap,:] = pole_params[swap][:, [4,5,6,7,  0,1,2,3]]
        
    elif pole_class == 5:
        #Order poles by Re(Pole-position)**2
        params1 = pole_params[:, 0:2]
        params2 = pole_params[:, 2:4]
        val1    = params1[:,0]**2
        val2    = params2[:,0]**2
        swap    = ~(val1 < val2)
        pole_params[swap,:] = pole_params[swap][:, [2,3,  0,1,  4,5]]

        params1 = pole_params[:, 0:2]
        params2 = pole_params[:, 4:6]
        val1    = params1[:,0]**2
        val2    = params2[:,0]**2
        swap    = ~(val1 < val2)
        pole_params[swap,:] = pole_params[swap][:, [4,5,  2,3,  0,1]]
        
        #Order poles by Re(Pole-position)**2
        params1 = pole_params[:, 2:4]
        params2 = pole_params[:, 4:6]
        val1    = params1[:,0]**2
        val2    = params2[:,0]**2
        swap    = ~(val1 < val2)
        pole_params[swap,:] = pole_params[swap][:, [0,1,  4,5,  2,3]]
        
    elif pole_class == 6:
        #make sure that Im(Pole-position)>0):
        indices = pole_params[:,5]    <  0
        pole_params[indices,5]       *= -1
        pole_params[indices,7]       *= -1
        
        #Order poles by Re(Pole-position)**2
        params1 = pole_params[:, 0:2]
        params2 = pole_params[:, 2:4]
        val1    = params1[:,0]**2
        val2    = params2[:,0]**2
        swap    = ~(val1 < val2)
        pole_params[swap,:] = pole_params[swap][:, [2,3,  0,1,  4,5,6,7]]
        
    elif pole_class == 7:
        #make sure that Im(Pole-position)>0):
        indices = pole_params[:,3]    <  0
        pole_params[indices,3]       *= -1
        pole_params[indices,5]       *= -1
        #make sure that Im(Pole-position)>0):
        indices = pole_params[:,7]    <  0
        pole_params[indices,7]       *= -1
        pole_params[indices,9]       *= -1
        
        #Order poles by Re(Pole-position)**2
        params1 = pole_params[:, 2:6]
        params2 = pole_params[:, 6:10]
        val1    = params1[:,0]**2
        val2    = params2[:,0]**2
        swap    = ~(val1 < val2)
        pole_params[swap,:] = pole_params[swap][:, [0,1,  6,7,8,9,  2,3,4,5]]
        
    elif pole_class == 8:
        #make sure that Im(Pole-position)>0):
        indices = pole_params[:,1]    <  0
        pole_params[indices,1]       *= -1
        pole_params[indices,3]       *= -1
        #make sure that Im(Pole-position)>0):
        indices = pole_params[:,5]    <  0
        pole_params[indices,5]       *= -1
        pole_params[indices,7]       *= -1
        #make sure that Im(Pole-position)>0):
        indices = pole_params[:,9]    <  0
        pole_params[indices,9]       *= -1
        pole_params[indices,11]       *= -1
        
        #Order poles by Re(Pole-position)**2
        params1 = pole_params[:, 0:4]
        params2 = pole_params[:, 4:8]
        val1    = params1[:,0]**2
        val2    = params2[:,0]**2
        swap    = ~(val1 < val2)
        pole_params[swap,:] = pole_params[swap][:, [4,5,6,7,  0,1,2,3,  8,9,10,11]]

        params1 = pole_params[:, 0:4]
        params2 = pole_params[:, 8:12]
        val1    = params1[:,0]**2
        val2    = params2[:,0]**2
        swap    = ~(val1 < val2)
        pole_params[swap,:] = pole_params[swap][:, [8,9,10,11,  4,5,6,7,  0,1,2,3]]
        
        #Order poles by Re(Pole-position)**2
        params1 = pole_params[:, 4:8]
        params2 = pole_params[:, 8:12]
        val1    = params1[:,0]**2
        val2    = params2[:,0]**2
        swap    = ~(val1 < val2)
        pole_params[swap,:] = pole_params[swap][:, [0,1,2,3,  8,9,10,11,  4,5,6,7]]
        
    return pole_params


def pole_config_organize_abs2(pole_class, pole_params):
    '''
    Organize pole configs such that:
        
        Parameters of real poles are always kept at the front
        
        Poles are ordered by Re(pole position)**2 + Im(pole position)**2 from small to large
        
        Im(Pole-position)>0 (convention, see parameters file)
        
    Note: Assumes poles to be ordered like in get_train_params(), but with imaginary parts of real poles removed (see pole_curve_calc vs pole_curve_calc2)
        
    pole_class: int = 0-8 
        The class of the pole configuration to be found
        
    pole_params: numpy.ndarray or torch.Tensor of shape (m,k), where m is the number of samples and k depends on the pole class (e.g. k=2 for pole_class=0)
        Pole configurations to be organized
        
    returns: numpy.ndarray or torch.Tensor of shape (m,k)
        The organized pole configurations
    '''
    if pole_class == 0:
        None
        
    elif pole_class == 1:
        #make sure that Im(Pole-position)>0):
        indices = pole_params[:,1]    <  0
        pole_params[indices,1]       *= -1
        pole_params[indices,3]       *= -1
        
    elif pole_class == 2:
        #Order poles by Re(Pole-position)**2
        params1 = pole_params[:, 0:2]
        params2 = pole_params[:, 2:4]
        val1    = params1[:,0]**2 
        val2    = params2[:,0]**2
        swap    = ~(val1 < val2)
        pole_params[swap,:] = pole_params[swap][:, [2,3,  0,1]]
        
    elif pole_class == 3:
        #make sure that Im(Pole-position)>0):
        indices = pole_params[:,3]    <  0
        pole_params[indices,3]       *= -1
        pole_params[indices,5]       *= -1
        
    elif pole_class == 4:
        #make sure that Im(Pole-position)>0):
        indices = pole_params[:,1]    <  0
        pole_params[indices,1]       *= -1
        pole_params[indices,3]       *= -1
        #make sure that Im(Pole-position)>0):
        indices = pole_params[:,5]    <  0
        pole_params[indices,5]       *= -1
        pole_params[indices,7]       *= -1
        
        #Order poles by Re(Pole-position)**2
        params1 = pole_params[:, 0:4]
        params2 = pole_params[:, 4:8]
        val1    = params1[:,0]**2 + params1[:,1]**2
        val2    = params2[:,0]**2 + params2[:,1]**2
        swap    = ~(val1 < val2)
        pole_params[swap,:] = pole_params[swap][:, [4,5,6,7,  0,1,2,3]]
        
    elif pole_class == 5:
        #Order poles by Re(Pole-position)**2
        params1 = pole_params[:, 0:2]
        params2 = pole_params[:, 2:4]
        val1    = params1[:,0]**2
        val2    = params2[:,0]**2
        swap    = ~(val1 < val2)
        pole_params[swap,:] = pole_params[swap][:, [2,3,  0,1,  4,5]]

        params1 = pole_params[:, 0:2]
        params2 = pole_params[:, 4:6]
        val1    = params1[:,0]**2
        val2    = params2[:,0]**2
        swap    = ~(val1 < val2)
        pole_params[swap,:] = pole_params[swap][:, [4,5,  2,3,  0,1]]
        
        #Order poles by Re(Pole-position)**2
        params1 = pole_params[:, 2:4]
        params2 = pole_params[:, 4:6]
        val1    = params1[:,0]**2
        val2    = params2[:,0]**2
        swap    = ~(val1 < val2)
        pole_params[swap,:] = pole_params[swap][:, [0,1,  4,5,  2,3]]
        
    elif pole_class == 6:
        #make sure that Im(Pole-position)>0):
        indices = pole_params[:,5]    <  0
        pole_params[indices,5]       *= -1
        pole_params[indices,7]       *= -1
        
        #Order poles by Re(Pole-position)**2
        params1 = pole_params[:, 0:2]
        params2 = pole_params[:, 2:4]
        val1    = params1[:,0]**2
        val2    = params2[:,0]**2
        swap    = ~(val1 < val2)
        pole_params[swap,:] = pole_params[swap][:, [2,3,  0,1,  4,5,6,7]]
        
    elif pole_class == 7:
        #make sure that Im(Pole-position)>0):
        indices = pole_params[:,3]    <  0
        pole_params[indices,3]       *= -1
        pole_params[indices,5]       *= -1
        #make sure that Im(Pole-position)>0):
        indices = pole_params[:,7]    <  0
        pole_params[indices,7]       *= -1
        pole_params[indices,9]       *= -1
        
        #Order poles by Re(Pole-position)**2
        params1 = pole_params[:, 2:6]
        params2 = pole_params[:, 6:10]
        val1    = params1[:,0]**2 + params1[:,1]**2
        val2    = params2[:,0]**2 + params2[:,1]**2
        swap    = ~(val1 < val2)
        pole_params[swap,:] = pole_params[swap][:, [0,1,  6,7,8,9,  2,3,4,5]]
        
    elif pole_class == 8:
        #make sure that Im(Pole-position)>0):
        indices = pole_params[:,1]    <  0
        pole_params[indices,1]       *= -1
        pole_params[indices,3]       *= -1
        #make sure that Im(Pole-position)>0):
        indices = pole_params[:,5]    <  0
        pole_params[indices,5]       *= -1
        pole_params[indices,7]       *= -1
        #make sure that Im(Pole-position)>0):
        indices = pole_params[:,9]    <  0
        pole_params[indices,9]       *= -1
        pole_params[indices,11]       *= -1
        
        #Order poles by Re(Pole-position)**2
        params1 = pole_params[:, 0:4]
        params2 = pole_params[:, 4:8]
        val1    = params1[:,0]**2 + params1[:,1]**2
        val2    = params2[:,0]**2 + params2[:,1]**2
        swap    = ~(val1 < val2)
        pole_params[swap,:] = pole_params[swap][:, [4,5,6,7,  0,1,2,3,  8,9,10,11]]

        params1 = pole_params[:, 0:4]
        params2 = pole_params[:, 8:12]
        val1    = params1[:,0]**2 + params1[:,1]**2
        val2    = params2[:,0]**2 + params2[:,1]**2
        swap    = ~(val1 < val2)
        pole_params[swap,:] = pole_params[swap][:, [8,9,10,11,  4,5,6,7,  0,1,2,3]]
        
        #Order poles by Re(Pole-position)**2
        params1 = pole_params[:, 4:8]
        params2 = pole_params[:, 8:12]
        val1    = params1[:,0]**2 + params1[:,1]**2
        val2    = params2[:,0]**2 + params2[:,1]**2
        swap    = ~(val1 < val2)
        pole_params[swap,:] = pole_params[swap][:, [0,1,2,3,  8,9,10,11,  4,5,6,7]]
        
    return pole_params




def pole_config_organize_abs(pole_class, pole_params):
    '''
    Organize pole configs such that:
        
        Poles are ordered by Re(Pole-position)**2 + Im(Position)**2 from small to large
        
        Im(Pole-position)>0 (convention, see parameters file)
        
    Note: Assumes poles to be ordered like in get_train_params()
        
    pole_class: int = 0-8 
        The class of the pole configuration to be found
        
    pole_params: numpy.ndarray or torch.Tensor of shape (m,k), where m is the number of samples and k depends on the pole class (e.g. k=2 for pole_class=0)
        Pole configurations to be organized
        
    returns: numpy.ndarray or torch.Tensor of shape (m,k)
        The organized pole configurations
    '''
    if pole_class == 0 or pole_class==1:
        #make sure that Im(Pole-position)>0):
        indices = pole_params[:,1]    <  0
        pole_params[indices,1]       *= -1
        pole_params[indices,3]       *= -1
        
    elif pole_class == 2 or pole_class==3 or pole_class==4:
        #make sure that Im(Pole-position)>0):
        indices = pole_params[:,1]    <  0
        pole_params[indices,1]       *= -1
        pole_params[indices,3]       *= -1
        #make sure that Im(Pole-position)>0):
        indices = pole_params[:,5]    <  0
        pole_params[indices,5]       *= -1
        pole_params[indices,7]       *= -1
        
        #Order poles by Re(Pole-position)**2 + Im(Position)**2
        params1 = pole_params[:, 0:4]
        params2 = pole_params[:, 4:8]
        val1    = params1[:,0]**2 + params1[:,1]**2
        val2    = params2[:,0]**2 + params2[:,1]**2
        swap    = ~(val1 < val2)
        pole_params[swap,:] = pole_params[swap][:, [4,5,6,7,  0,1,2,3]]
        
    elif pole_class == 5 or pole_class == 6 or pole_class == 7 or pole_class == 8:  
        #make sure that Im(Pole-position)>0):
        indices = pole_params[:,1]    <  0
        pole_params[indices,1]       *= -1
        pole_params[indices,3]       *= -1
        #make sure that Im(Pole-position)>0):
        indices = pole_params[:,5]    <  0
        pole_params[indices,5]       *= -1
        pole_params[indices,7]       *= -1
        #make sure that Im(Pole-position)>0):
        indices = pole_params[:,9]    <  0
        pole_params[indices,9]       *= -1
        pole_params[indices,11]       *= -1
        
        #Order poles by Re(Pole-position)**2 + Im(Position)**2
        params1 = pole_params[:, 0:4]
        params2 = pole_params[:, 4:8]
        val1    = params1[:,0]**2 + params1[:,1]**2
        val2    = params2[:,0]**2 + params2[:,1]**2
        swap    = ~(val1 < val2)
        pole_params[swap,:] = pole_params[swap][:, [4,5,6,7,  0,1,2,3,  8,9,10,11]]

        params1 = pole_params[:, 0:4]
        params2 = pole_params[:, 8:12]
        val1    = params1[:,0]**2 + params1[:,1]**2
        val2    = params2[:,0]**2 + params2[:,1]**2
        swap    = ~(val1 < val2)
        pole_params[swap,:] = pole_params[swap][:, [8,9,10,11,  4,5,6,7,  0,1,2,3]]
        
        #Order poles by Re(Pole-position)**2
        params1 = pole_params[:, 4:8]
        params2 = pole_params[:, 8:12]
        val1    = params1[:,0]**2 + params1[:,1]**2
        val2    = params2[:,0]**2 + params2[:,1]**2
        swap    = ~(val1 < val2)
        pole_params[swap,:] = pole_params[swap][:, [0,1,2,3,  8,9,10,11,  4,5,6,7]]

    return pole_params


def pole_config_organize_re(pole_class, pole_params):
    '''
    Organize pole configs such that:
        
        Poles are ordered by Re(Pole-position)**2  from small to large
        
        Im(Pole-position)>0 (convention, see parameters file)
        
    Note: Assumes poles to be ordered like in get_train_params()
        
    pole_class: int = 0-8 
        The class of the pole configuration to be found
        
    pole_params: numpy.ndarray or torch.Tensor of shape (m,k), where m is the number of samples and k depends on the pole class (e.g. k=2 for pole_class=0)
        Pole configurations to be organized
        
    returns: numpy.ndarray or torch.Tensor of shape (m,k)
        The organized pole configurations
    '''
    if pole_class == 0 or pole_class==1:
        #make sure that Im(Pole-position)>0):
        indices = pole_params[:,1]    <  0
        pole_params[indices,1]       *= -1
        pole_params[indices,3]       *= -1
        
    elif pole_class == 2 or pole_class==3 or pole_class==4:
        #make sure that Im(Pole-position)>0):
        indices = pole_params[:,1]    <  0
        pole_params[indices,1]       *= -1
        pole_params[indices,3]       *= -1
        #make sure that Im(Pole-position)>0):
        indices = pole_params[:,5]    <  0
        pole_params[indices,5]       *= -1
        pole_params[indices,7]       *= -1
        
        #Order poles by Re(Pole-position)**2 + Im(Position)**2
        params1 = pole_params[:, 0:4]
        params2 = pole_params[:, 4:8]
        val1    = params1[:,0]**2 
        val2    = params2[:,0]**2 
        swap    = ~(val1 < val2)
        pole_params[swap,:] = pole_params[swap][:, [4,5,6,7,  0,1,2,3]]
        
    elif pole_class == 5 or pole_class == 6 or pole_class == 7 or pole_class == 8:  
        #make sure that Im(Pole-position)>0):
        indices = pole_params[:,1]    <  0
        pole_params[indices,1]       *= -1
        pole_params[indices,3]       *= -1
        #make sure that Im(Pole-position)>0):
        indices = pole_params[:,5]    <  0
        pole_params[indices,5]       *= -1
        pole_params[indices,7]       *= -1
        #make sure that Im(Pole-position)>0):
        indices = pole_params[:,9]    <  0
        pole_params[indices,9]       *= -1
        pole_params[indices,11]       *= -1
        
        #Order poles by Re(Pole-position)**2 + Im(Position)**2
        params1 = pole_params[:, 0:4]
        params2 = pole_params[:, 4:8]
        val1    = params1[:,0]**2 
        val2    = params2[:,0]**2 
        swap    = ~(val1 < val2)
        pole_params[swap,:] = pole_params[swap][:, [4,5,6,7,  0,1,2,3,  8,9,10,11]]

        params1 = pole_params[:, 0:4]
        params2 = pole_params[:, 8:12]
        val1    = params1[:,0]**2 
        val2    = params2[:,0]**2 
        swap    = ~(val1 < val2)
        pole_params[swap,:] = pole_params[swap][:, [8,9,10,11,  4,5,6,7,  0,1,2,3]]
        
        #Order poles by Re(Pole-position)**2
        params1 = pole_params[:, 4:8]
        params2 = pole_params[:, 8:12]
        val1    = params1[:,0]**2 
        val2    = params2[:,0]**2 
        swap    = ~(val1 < val2)
        pole_params[swap,:] = pole_params[swap][:, [0,1,2,3,  8,9,10,11,  4,5,6,7]]

    return pole_params








