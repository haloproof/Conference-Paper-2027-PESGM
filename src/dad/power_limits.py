"""AC decision/settlement power, with explicit optional battery-terminal DC cap."""


def ac_to_dc_kw(power, config):
    # Both sides use positive discharge here; native SimSES uses positive charge.
    return power/config['eta_discharge'] if power>=0 else power*config['eta_charge']


def ac_power_bounds(config):
    if 'dc_power_kw' in config:
        limit=config['dc_power_kw']
        return -limit/config['eta_charge'],limit*config['eta_discharge']
    return -config['power_kw'],config['power_kw']  # Legacy AC configurations.


def power_span_kw(config):
    lo,hi=ac_power_bounds(config)
    return hi-lo


def power_margin(command_ac, delivered_dc, config):
    if 'dc_power_kw' in config:
        return min(config['dc_power_kw']-abs(ac_to_dc_kw(command_ac,config)),
                   config['dc_power_kw']-abs(delivered_dc))
    return config['power_kw']-abs(command_ac)
