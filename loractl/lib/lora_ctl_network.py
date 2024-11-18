from modules import extra_networks, script_callbacks, shared
from loractl.lib import utils
import ldm_patched.modules.sd
import ldm_patched.modules.utils
import sys, os
from pathlib import Path

# extensions-builtin isn't normally referencable due to the dash; this hacks around that
lora_path = str(Path(__file__).parent.parent.parent.parent.parent / "extensions-builtin" / "Lora")
sys.path.insert(0, lora_path)
import network, networks, extra_networks_lora
sys.path.remove(lora_path)

lora_weights = {}

def reset_weights():
    lora_weights.clear()

def find_lora_path(name):
    """Search for LoRA file in main directory and subdirectories"""
    base_path = Path(shared.cmd_opts.lora_dir)
    
    # List of possible extensions
    extensions = ['.safetensors', '.ckpt', '.pt']
    
    # First check direct path with extensions
    for ext in extensions:
        direct_path = base_path / f"{name}{ext}"
        if direct_path.exists():
            return direct_path
            
    # Search in subdirectories
    for subdir in base_path.rglob("*"):
        if subdir.is_dir():
            for ext in extensions:
                path = subdir / f"{name}{ext}"
                if path.exists():
                    return path
                    
    return None

class LoraCtlNetwork(extra_networks.ExtraNetwork):
    def __init__(self):
        super().__init__('lora')
        self.errors = {}
        self.current_processing = None
        self.loaded_loras = {}

    def activate(self, p, params_list):
        if not utils.is_active():
            return super().activate(p, params_list)

        self.current_processing = p

        for params in params_list:
            assert params.items
            name = params.positional[0]
            print(f"Processing LoRA {name} with params: {params}")
            
            if lora_weights.get(name, None) is None:
                weights = utils.params_to_weights(params)
                print(f"Calculated weights for {name}: {weights}")
                lora_weights[name] = weights

        # Get current model state
        current_sd = p.sd_model
        if current_sd is None:
            return

        # Store original state for deactivation
        if not hasattr(current_sd, 'original_network_state'):
            current_sd.original_network_state = {
                'unet': current_sd.forge_objects.unet,
                'clip': current_sd.forge_objects.clip,
                'lora_names': [],
                'lora_weights': {}
            }

        names = []
        unet_weights = []
        te_weights = []

        # Calculate initial weights for each LoRA
        for name, weights in lora_weights.items():
            lora_path = find_lora_path(name)
            if lora_path is None:
                print(f"LoRA file not found for: {name}")
                continue

            names.append(name)
            initial_unet_weight = utils.calculate_weight(
                weights["unet"],
                0,  # initial step
                p.steps,
                step_offset=2
            )
            initial_te_weight = utils.calculate_weight(
                weights["te"],
                0,  # initial step
                p.steps,
                step_offset=2
            )
            unet_weights.append(initial_unet_weight)
            te_weights.append(initial_te_weight)
            print(f"Initial weights for {name} - UNet: {initial_unet_weight}, TE: {initial_te_weight}")

        # Load networks using the base implementation
        networks.load_networks(names, te_weights, unet_weights)

        # Register callback for weight updates during sampling
        def cfg_callback(params: script_callbacks.CFGDenoiserParams):
            if not lora_weights or not hasattr(self, 'current_processing'):
                return

            step = params.sampling_step
            total_steps = params.total_sampling_steps

            updated_te_weights = []
            updated_unet_weights = []

            for name, weights in lora_weights.items():
                current_unet_weight = utils.calculate_weight(
                    weights["unet"],
                    step,
                    total_steps,
                    step_offset=2
                )
                current_te_weight = utils.calculate_weight(
                    weights["te"],
                    step,
                    total_steps,
                    step_offset=2
                )
                updated_unet_weights.append(current_unet_weight)
                updated_te_weights.append(current_te_weight)
                print(f"Step {step}/{total_steps} - {name} weights - UNet: {current_unet_weight}, TE: {current_te_weight}")

            # Update network weights
            networks.load_networks(names, updated_te_weights, updated_unet_weights)

        script_callbacks.on_cfg_denoiser(cfg_callback)

    def deactivate(self, p):
        """Reset model state and clear weights after generation"""
        if hasattr(p, 'sd_model') and hasattr(p.sd_model, 'original_network_state'):
            current_sd = p.sd_model
            current_sd.forge_objects.unet = current_sd.original_network_state['unet']
            current_sd.forge_objects.clip = current_sd.original_network_state['clip']
            delattr(current_sd, 'original_network_state')
        
        self.current_processing = None
        self.loaded_loras.clear()
        
        reset_weights()
        
        if self.errors:
            p.comment("Networks with errors: " + ", ".join(f"{k} ({v})" for k, v in self.errors.items()))
            self.errors.clear()