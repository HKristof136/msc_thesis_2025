def price_function(pricer, points=None):
    if points is not None:
        return pricer.price(points)
    else:
        return pricer.price()

def delta_function(pricer, points=None):
    if points is not None:
        return pricer.delta(points)
    else:
        return pricer.delta()

def gamma_function(pricer, points=None):
    if points is not None:
        return pricer.gamma(points)
    else:
        return pricer.gamma()

def vega_function(pricer, points=None):
    if points is not None:
        return pricer.vega(points)
    else:
        return pricer.vega()

def theta_function(pricer, points=None):
    if points is not None:
        return pricer.theta(points)
    else:
        return pricer.theta()

def rho_function(pricer, points=None):
    if points is not None:
        return pricer.rho(points)
    else:
        return pricer.rho()

def partial_deriv_correlation_function(pricer, points=None):
    if points is not None:
        return pricer.partial_deriv_correlation(points)
    else:
        return pricer.partial_deriv_correlation()

def partial_deriv_kappa_function(pricer, points=None):
    if points is not None:
        return pricer.partial_deriv_kappa(points)
    else:
        return pricer.partial_deriv_kappa()

def partial_deriv_theta_function(pricer, points=None):
    if points is not None:
        return pricer.partial_deriv_theta(points)
    else:
        return pricer.partial_deriv_theta()

def partial_deriv_sigma_function(pricer, points=None):
    if points is not None:
        return pricer.partial_deriv_sigma(points)
    else:
        return pricer.partial_deriv_sigma()
