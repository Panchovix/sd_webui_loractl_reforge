import numpy as np
import re

# Given a string like x@y,z@a, returns [[x, z], [y, a]] sorted for consumption by np.interp


def sorted_positions(raw_steps):
    steps = [[float(s.strip()) for s in re.split("[@~]", x)]
             for x in re.split("[,;]", str(raw_steps))]
    # If we just got a single number, just return it
    if len(steps[0]) == 1:
        return steps[0][0]

    # Add implicit 1s to any steps which don't have a weight
    steps = [[s[0], s[1] if len(s) == 2 else 1] for s in steps]

    # Sort by index
    steps.sort(key=lambda k: k[1])

    steps = [list(v) for v in zip(*steps)]
    return steps


def calculate_weight(m, step, max_steps, step_offset=2):
    if isinstance(m, (int, float)):
        return float(m)
        
    if isinstance(m, list):
        # Convert step to progress percentage (0-1)
        if max_steps > 0:
            progress = step / (max_steps - step_offset)
        else:
            progress = 1.0
            
        # If control points are normalized (0-1)
        if all(p <= 1.0 for p in m[1]):
            # Use progress as is
            t = progress
        else:
            # Convert progress to step scale
            t = progress * max_steps
            
        # Ensure we have proper control points
        weights, positions = m
        if len(weights) != len(positions):
            return weights[0]
            
        # Find the bracketing control points
        for i in range(len(positions)-1):
            if positions[i] <= t <= positions[i+1]:
                # Calculate interpolation factor
                factor = (t - positions[i]) / (positions[i+1] - positions[i])
                # Linearly interpolate between weights
                return weights[i] + factor * (weights[i+1] - weights[i])
                
        # If we're before first point, return first weight
        if t <= positions[0]:
            return weights[0]
        # If we're after last point, return last weight
        return weights[-1]
        
    return 1.0  # Default weight if no valid specification


def params_to_weights(params):
    weights = {
        "unet": None,  # UNet weights
        "te": 1.0,    # Text encoder weights
        "hrunet": None,  # High-res fix UNet weights
        "hrte": None,   # High-res fix text encoder weights
        "block_weights": None,  # Optional block weights
        "block_ranges": None    # Optional block weight ranges
    }

    # Handle positional arguments
    if len(params.positional) > 1:
        weights["te"] = sorted_positions(params.positional[1])

    if len(params.positional) > 2:
        weights["unet"] = sorted_positions(params.positional[2])
        
    # Handle block weights if present in third positional argument
    if len(params.positional) > 3:
        try:
            block_weight_text = str(params.positional[3])
            if ";" in block_weight_text or "=" in block_weight_text:
                block_weights = {}
                block_ranges = {}
                
                # Parse block weight specifications
                for block_spec in block_weight_text.split(";"):
                    if not block_spec.strip():
                        continue
                        
                    if "=" not in block_spec:
                        continue
                        
                    block_name, weight_spec = block_spec.split("=", 1)
                    block_name = block_name.strip().upper()
                    
                    # Handle range specifications like "IN01-OUT11"
                    if "-" in block_name:
                        start_block, end_block = block_name.split("-", 1)
                        block_ranges[start_block] = (end_block, float(weight_spec))
                    else:
                        block_weights[block_name] = float(weight_spec)
                        
                if block_weights:
                    weights["block_weights"] = block_weights
                if block_ranges:
                    weights["block_ranges"] = block_ranges
        except Exception as e:
            print(f"Error parsing block weights: {e}")

    # Handle named parameters - these override positional ones
    if params.named.get("te"):
        weights["te"] = sorted_positions(params.named.get("te"))

    if params.named.get("unet"):
        weights["unet"] = sorted_positions(params.named.get("unet"))

    # Handle high-res fix weights
    if params.named.get("hr"):
        # If single hr value provided, use for both unet and te
        hr_weight = sorted_positions(params.named.get("hr"))
        weights["hrunet"] = hr_weight
        weights["hrte"] = hr_weight

    if params.named.get("hrunet"):
        weights["hrunet"] = sorted_positions(params.named.get("hrunet"))

    if params.named.get("hrte"):
        weights["hrte"] = sorted_positions(params.named.get("hrte"))

    # Handle block weights from named parameters
    if params.named.get("blocks"):
        try:
            block_weights = {}
            block_ranges = {}
            block_spec = str(params.named.get("blocks"))
            
            for block_def in block_spec.split(";"):
                if not block_def.strip():
                    continue
                    
                if "=" not in block_def:
                    continue
                    
                block_name, weight_spec = block_def.split("=", 1)
                block_name = block_name.strip().upper()
                
                if "-" in block_name:
                    start_block, end_block = block_name.split("-", 1)
                    block_ranges[start_block] = (end_block, float(weight_spec))
                else:
                    block_weights[block_name] = float(weight_spec)
                    
            if block_weights:
                weights["block_weights"] = block_weights
            if block_ranges:
                weights["block_ranges"] = block_ranges
        except Exception as e:
            print(f"Error parsing named block weights: {e}")

    # If unet weight is not set, use text encoder weight
    weights["unet"] = weights["unet"] if weights["unet"] is not None else weights["te"]
    
    # If high-res fix weights not set, use regular weights
    weights["hrunet"] = weights["hrunet"] if weights["hrunet"] is not None else weights["unet"]
    weights["hrte"] = weights["hrte"] if weights["hrte"] is not None else weights["te"]

    return weights


hires = False
loractl_active = True

def is_hires():
    return hires


def set_hires(value):
    global hires
    hires = value


def set_active(value):
    global loractl_active
    loractl_active = value

def is_active():
    global loractl_active
    return loractl_active
