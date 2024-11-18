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

def load_lora_weights(filename):
    """Load weights from a LoRA file"""
    print(f"Loading LoRA weights from: {filename}")
    
    # First load the raw state dict
    state_dict = ldm_patched.modules.utils.load_torch_file(filename, safe_load=True)
    
    # Split the state dict into CLIP and UNET parts
    clip_weights = {}
    unet_weights = {}
    
    for key, value in state_dict.items():
        if key.startswith('lora_te_'):
            clip_weights[key] = value
        else:
            unet_weights[key] = value
            
    return {
        'unet': unet_weights,
        'clip': clip_weights,
        'name': os.path.basename(filename)
    }

class LoraCtlNetwork(extra_networks.ExtraNetwork):
    def __init__(self):
        super().__init__('lora')
        self.errors = {}
        self.current_processing = None
        self.loaded_loras = {}

    def activate(self, p, params_list):
        if not utils.is_active():
            return super().activate(p, params_list)

        # Store processing object for later use
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
        if not hasattr(current_sd, 'original_unet'):
            current_sd.original_unet = current_sd.forge_objects.unet
            current_sd.original_clip = current_sd.forge_objects.clip

        # Reset to original state before applying new weights
        current_sd.forge_objects.unet = current_sd.original_unet
        current_sd.forge_objects.clip = current_sd.original_clip

        for name, weights in lora_weights.items():
            # Find LoRA file
            lora_path = find_lora_path(name)
            if lora_path is None:
                print(f"LoRA file not found for: {name}")
                continue

            try:
                # Load and cache LoRA weights
                if str(lora_path) not in self.loaded_loras:
                    self.loaded_loras[str(lora_path)] = load_lora_weights(str(lora_path))
                lora_data = self.loaded_loras[str(lora_path)]
                
                # Calculate initial weights
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
                
                print(f"Initial weights for {name} - UNet: {initial_unet_weight}, TE: {initial_te_weight}")

                # Apply initial weights
                current_sd.forge_objects.unet, current_sd.forge_objects.clip = ldm_patched.modules.sd.load_lora_for_models(
                    current_sd.forge_objects.unet,
                    current_sd.forge_objects.clip,
                    lora_data['unet'],  # Only pass the UNET weights
                    initial_unet_weight,
                    initial_te_weight,
                    filename=name
                )

            except Exception as e:
                self.errors[name] = str(e)
                print(f"Error applying LoRA {name}: {e}")
                import traceback
                traceback.print_exc()
                continue

        # Register callback for the CFGDenoiser
        def cfg_callback(params: script_callbacks.CFGDenoiserParams):
            if not lora_weights or not hasattr(self, 'current_processing'):
                return

            current_sd = self.current_processing.sd_model
            step = params.sampling_step
            total_steps = params.total_sampling_steps

            # Reset to original state before applying new weights
            current_sd.forge_objects.unet = current_sd.original_unet
            current_sd.forge_objects.clip = current_sd.original_clip

            for name, weights in lora_weights.items():
                try:
                    # Get cached LoRA data
                    lora_path = find_lora_path(name)
                    if lora_path is None:
                        continue

                    lora_data = self.loaded_loras.get(str(lora_path))
                    if lora_data is None:
                        continue

                    # Calculate weights for current step
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
                    
                    print(f"Step {step}/{total_steps} - {name} weights - UNet: {current_unet_weight}, TE: {current_te_weight}")
                    
                    # Apply LoRA with new weights
                    current_sd.forge_objects.unet, current_sd.forge_objects.clip = ldm_patched.modules.sd.load_lora_for_models(
                        current_sd.forge_objects.unet,
                        current_sd.forge_objects.clip,
                        lora_data['unet'],  # Only pass the UNET weights
                        current_unet_weight,
                        current_te_weight,
                        filename=name
                    )

                except Exception as e:
                    print(f"Error updating LoRA {name} at step {step}: {e}")
                    import traceback
                    traceback.print_exc()

        # Register the callback
        script_callbacks.on_cfg_denoiser(cfg_callback)

    def deactivate(self, p):
        """Reset model state and clear weights after generation"""
        if hasattr(p, 'sd_model'):
            current_sd = p.sd_model
            if hasattr(current_sd, 'original_unet'):
                current_sd.forge_objects.unet = current_sd.original_unet
                current_sd.forge_objects.clip = current_sd.original_clip
        
        # Clear current processing reference and caches
        self.current_processing = None
        self.loaded_loras.clear()
        
        reset_weights()
        
        if self.errors:
            p.comment("Networks with errors: " + ", ".join(f"{k} ({v})" for k, v in self.errors.items()))
            self.errors.clear()