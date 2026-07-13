#GPT/utils/loss_debugging.py
"""
Loss Spike Diagnosis Tools
===========================

Comprehensive diagnostics for the 5 loss spike causes:

1. Gradient Explosion - huge gradient norms
2. Learning Rate Too High - loss bouncing
3. Data Quality Issue - NaN on specific batches
4. Activation Explosion - hidden layer values growing huge
5. Checkpoint Loading Error - spike after loading weights

Usage:
    from GPT.utils.loss_debugging import diagnose_loss_spike
    
    # During training, if loss spikes:
    report = diagnose_loss_spike(
        model=model,
        batch=batch,
        loss=loss_value,
        grad_norm=grad_norm_value,
        optimizer=optimizer
    )
    
    if report["status"] != "ok":
        print(f"⚠ Issues detected: {report['likely_causes']}")
        print(f"Recommendations: {report['recommendations']}")
"""

import tensorflow as tf
import numpy as np
from typing import Dict, List, Optional, Tuple
import json


class LossDebugger:
    """Tools for diagnosing loss spikes during training."""
    
    @staticmethod
    def check_gradient_explosion(grad_norm: float, threshold: float = 100.0) -> Dict:
        """Check if gradient norm indicates explosion.
        
        Args:
            grad_norm: Current gradient norm (already computed)
            threshold: Threshold for explosion (> 100 is usually bad)
        
        Returns:
            Dictionary with status and details
        """
        return {
            "grad_norm": float(grad_norm),
            "is_exploding": float(grad_norm) > threshold,
            "severity": "critical" if float(grad_norm) > 1000 else "warning" if float(grad_norm) > threshold else "ok",
            "description": {
                "critical": "Gradient explosion critical - likely NaN incoming",
                "warning": "Gradient norm high - possible explosion",
                "ok": "Gradient norm in normal range",
            }.get("critical" if float(grad_norm) > 1000 else "warning" if float(grad_norm) > threshold else "ok")
        }
    
    @staticmethod
    def check_learning_rate(optimizer, threshold_high: float = 1e-3, threshold_low: float = 1e-5) -> Dict:
        """Check if learning rate is reasonable.
        
        Args:
            optimizer: TensorFlow optimizer with learning_rate
            threshold_high: LR above this is likely too high
            threshold_low: LR below this is likely too low
        
        Returns:
            Dictionary with status and details
        """
        try:
            lr = optimizer.learning_rate
            
            # Handle different LR types
            if hasattr(lr, 'numpy'):
                lr_val = float(lr.numpy())
            elif callable(lr):
                lr_val = float(lr(optimizer.iterations).numpy())
            else:
                lr_val = float(lr)
        except:
            lr_val = None
        
        if lr_val is None:
            return {
                "learning_rate": None,
                "status": "unknown",
                "description": "Could not determine learning rate"
            }
        
        if lr_val > threshold_high:
            severity = "warning"
            description = f"Learning rate {lr_val:.2e} is very high (> {threshold_high:.2e})"
        elif lr_val < threshold_low:
            severity = "warning"
            description = f"Learning rate {lr_val:.2e} is very low (< {threshold_low:.2e})"
        else:
            severity = "ok"
            description = f"Learning rate {lr_val:.2e} is in normal range"
        
        return {
            "learning_rate": lr_val,
            "status": severity,
            "threshold_high": threshold_high,
            "threshold_low": threshold_low,
            "description": description
        }
    
    @staticmethod
    def check_batch_statistics(batch: tf.Tensor) -> Dict:
        """Check batch statistics for data quality issues.
        
        Args:
            batch: Input batch tensor
        
        Returns:
            Dictionary with batch statistics
        """
        batch_np = batch.numpy()
        
        return {
            "shape": batch_np.shape,
            "dtype": str(batch.dtype),
            "mean": float(np.mean(batch_np)),
            "std": float(np.std(batch_np)),
            "min": float(np.min(batch_np)),
            "max": float(np.max(batch_np)),
            "has_nan": bool(np.any(np.isnan(batch_np))),
            "has_inf": bool(np.any(np.isinf(batch_np))),
            "vocab_range_ok": float(np.min(batch_np)) >= 0 and float(np.max(batch_np)) < 50257,
        }
    
    @staticmethod
    def check_loss_value(loss: float) -> Dict:
        """Check if loss value is valid.
        
        Args:
            loss: Loss value
        
        Returns:
            Dictionary with loss status
        """
        loss_val = float(loss)
        
        if np.isnan(loss_val):
            return {
                "loss": loss_val,
                "status": "critical",
                "is_nan": True,
                "is_inf": False,
                "description": "Loss is NaN - training diverged"
            }
        
        if np.isinf(loss_val):
            return {
                "loss": loss_val,
                "status": "critical",
                "is_nan": False,
                "is_inf": True,
                "description": "Loss is Inf - numerical overflow"
            }
        
        if loss_val > 1000:
            return {
                "loss": loss_val,
                "status": "warning",
                "is_nan": False,
                "is_inf": False,
                "description": "Loss is extremely high (> 1000)"
            }
        
        return {
            "loss": loss_val,
            "status": "ok",
            "is_nan": False,
            "is_inf": False,
            "description": "Loss is valid"
        }
    
    @staticmethod
    def check_model_weights(model, sample_size: int = 3) -> Dict:
        """Check weight statistics for initialization issues.
        
        Args:
            model: GPT model
            sample_size: Number of layers to sample
        
        Returns:
            Dictionary with weight statistics
        """
        weight_stats = {}
        
        vars_to_check = model.trainable_variables[:sample_size]
        
        for var in vars_to_check:
            w_val = var.numpy()
            weight_stats[var.name] = {
                "shape": w_val.shape,
                "mean": float(np.mean(w_val)),
                "std": float(np.std(w_val)),
                "min": float(np.min(w_val)),
                "max": float(np.max(w_val)),
                "has_nan": bool(np.any(np.isnan(w_val))),
                "has_inf": bool(np.any(np.isinf(w_val))),
            }
        
        return {
            "num_vars_checked": sample_size,
            "total_vars": len(model.trainable_variables),
            "weights": weight_stats
        }
    
    @staticmethod
    def check_activations(model, batch: tf.Tensor) -> Dict:
        """Check activation statistics for explosion.
        
        Args:
            model: GPT model (must support return_hidden=True or similar)
            batch: Input batch
        
        Returns:
            Dictionary with activation statistics
        """
        try:
            # Try to get hidden states if model supports it
            # This assumes model has a return_hidden parameter
            if hasattr(model, 'call'):
                # Check if model can return hidden states
                try:
                    # Try with return_hidden parameter
                    output = model(batch, return_hidden=True)
                    if isinstance(output, dict):
                        hidden_states = output
                    else:
                        # Model doesn't return hidden states
                        return {
                            "status": "not_supported",
                            "message": "Model doesn't support hidden state inspection"
                        }
                except TypeError:
                    # return_hidden not supported
                    return {
                        "status": "not_supported",
                        "message": "Model doesn't support return_hidden parameter"
                    }
            else:
                return {
                    "status": "not_supported",
                    "message": "Cannot inspect activations"
                }
            
            activation_stats = {}
            max_activation_overall = 0.0
            has_explosion = False
            
            for layer_name, activation in hidden_states.items():
                if activation is None:
                    continue
                
                act_val = activation.numpy()
                mean_val = float(np.mean(act_val))
                std_val = float(np.std(act_val))
                max_val = float(np.max(np.abs(act_val)))
                
                max_activation_overall = max(max_activation_overall, max_val)
                
                is_exploding = max_val > 1e3
                if is_exploding:
                    has_explosion = True
                
                activation_stats[layer_name] = {
                    "mean": mean_val,
                    "std": std_val,
                    "max": max_val,
                    "has_nan": bool(np.any(np.isnan(act_val))),
                    "has_inf": bool(np.any(np.isinf(act_val))),
                    "is_exploding": is_exploding,
                }
            
            return {
                "status": "critical" if has_explosion else "ok",
                "max_activation": max_activation_overall,
                "layers": activation_stats,
                "has_explosion": has_explosion,
            }
        
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error checking activations: {str(e)}"
            }


def diagnose_loss_spike(
    model,
    batch: tf.Tensor,
    loss: float,
    grad_norm: float,
    optimizer,
) -> Dict:
    """Comprehensive diagnosis of loss spike causes.
    
    Checks all 5 possible causes and provides recommendations.
    
    Args:
        model: GPT model
        batch: Current training batch
        loss: Current loss value
        grad_norm: Current gradient norm
        optimizer: Optimizer instance
    
    Returns:
        Comprehensive diagnostic report with:
        - status: "ok", "warning", or "critical"
        - likely_causes: List of probable causes (1-5)
        - findings: Detailed findings for each check
        - recommendations: Actionable recommendations
        - scores: Numerical scores for each cause (0-100)
    """
    
    debugger = LossDebugger()
    
    # Run all checks
    print("\n" + "="*70)
    print("LOSS SPIKE DIAGNOSIS")
    print("="*70)
    
    print("\n[1/5] Checking loss value...")
    loss_check = debugger.check_loss_value(loss)
    print(f"  Loss: {loss_check['loss']:.6f}")
    print(f"  Status: {loss_check['status'].upper()}")
    
    print("\n[2/5] Checking gradient norms...")
    grad_check = debugger.check_gradient_explosion(grad_norm)
    print(f"  Grad norm: {grad_check['grad_norm']:.4f}")
    print(f"  Severity: {grad_check['severity'].upper()}")
    
    print("\n[3/5] Checking learning rate...")
    lr_check = debugger.check_learning_rate(optimizer)
    if lr_check['learning_rate'] is not None:
        print(f"  LR: {lr_check['learning_rate']:.2e}")
        print(f"  Status: {lr_check['status'].upper()}")
    else:
        print(f"  LR: Could not determine")
    
    print("\n[4/5] Checking batch statistics...")
    batch_check = debugger.check_batch_statistics(batch)
    print(f"  Shape: {batch_check['shape']}")
    print(f"  Mean: {batch_check['mean']:.4f}, Std: {batch_check['std']:.4f}")
    print(f"  Has NaN: {batch_check['has_nan']}, Has Inf: {batch_check['has_inf']}")
    print(f"  Vocab range OK: {batch_check['vocab_range_ok']}")
    
    print("\n[5/5] Checking activations...")
    activation_check = debugger.check_activations(model, batch)
    if activation_check.get('status') == 'not_supported':
        print(f"  {activation_check['message']}")
    elif activation_check.get('status') == 'error':
        print(f"  Error: {activation_check['message']}")
    else:
        print(f"  Max activation: {activation_check.get('max_activation', 0):.2e}")
        print(f"  Has explosion: {activation_check.get('has_explosion', False)}")
    
    # Compute likelihood scores for each cause (0-100)
    cause_scores = {
        "1_gradient_explosion": 0,
        "2_learning_rate_high": 0,
        "3_data_quality": 0,
        "4_activation_explosion": 0,
        "5_checkpoint_error": 0,
    }
    
    # Cause 1: Gradient Explosion
    if grad_check['severity'] == "critical":
        cause_scores["1_gradient_explosion"] = 95
    elif grad_check['severity'] == "warning":
        cause_scores["1_gradient_explosion"] = 60
    elif loss_check['status'] in ["critical", "warning"]:
        cause_scores["1_gradient_explosion"] = 40
    
    # Cause 2: Learning Rate Too High
    if lr_check['status'] == "warning" and lr_check['learning_rate'] and lr_check['learning_rate'] > 1e-3:
        cause_scores["2_learning_rate_high"] = 80
    elif lr_check['status'] == "warning":
        cause_scores["2_learning_rate_high"] = 50
    
    # Cause 3: Data Quality Issue
    if batch_check['has_nan'] or batch_check['has_inf']:
        cause_scores["3_data_quality"] = 90
    elif not batch_check['vocab_range_ok']:
        cause_scores["3_data_quality"] = 85
    elif loss_check['is_nan'] or loss_check['is_inf']:
        cause_scores["3_data_quality"] = 70
    
    # Cause 4: Activation Explosion
    if activation_check.get('has_explosion'):
        cause_scores["4_activation_explosion"] = 85
    elif activation_check.get('max_activation', 0) > 1e2:
        cause_scores["4_activation_explosion"] = 60
    
    # Cause 5: Checkpoint Error (harder to detect, usually happens right after load)
    # This is more about context than current batch
    cause_scores["5_checkpoint_error"] = 20  # Default low
    
    # Determine overall status
    if any(check['status'] == 'critical' for check in [loss_check, grad_check, batch_check]):
        overall_status = "critical"
    elif any(check['status'] == 'warning' for check in [loss_check, grad_check, batch_check, lr_check]):
        overall_status = "warning"
    else:
        overall_status = "ok"
    
    # Get likely causes (score > 50)
    likely_causes = []
    for cause, score in sorted(cause_scores.items(), key=lambda x: x[1], reverse=True):
        if score > 50:
            cause_name = cause.split('_', 1)[1].replace('_', ' ').title()
            likely_causes.append(f"Cause {cause.split('_')[0]}: {cause_name} (score: {score})")
    
    # Generate recommendations
    recommendations = []
    
    if cause_scores["1_gradient_explosion"] > 60:
        recommendations.append("✗ Gradient explosion likely")
        recommendations.append("  → Increase grad_clip_norm (try 0.5 instead of 1.0)")
        recommendations.append("  → Reduce learning_rate by 10x")
        recommendations.append("  → Check Layer Normalization is working")
    
    if cause_scores["2_learning_rate_high"] > 50:
        recommendations.append("✗ Learning rate may be too high")
        recommendations.append("  → Reduce learning_rate: try 3e-5 instead of 3e-4")
        recommendations.append("  → Increase warmup_steps: try 5000 instead of 2000")
    
    if cause_scores["3_data_quality"] > 50:
        recommendations.append("✗ Data quality issue likely")
        recommendations.append("  → Verify batch: check tokens are 0-50257")
        recommendations.append("  → Regenerate TFRecords if corrupted")
        recommendations.append("  → Check tokenizer version consistency")
    
    if cause_scores["4_activation_explosion"] > 50:
        recommendations.append("✗ Activation explosion likely")
        recommendations.append("  → Check LayerNorm epsilon (should be 1e-6 or higher)")
        recommendations.append("  → Verify pre-norm structure (LN before attention/MLP)")
        recommendations.append("  → Try reducing batch_size")
    
    if cause_scores["5_checkpoint_error"] > 50:
        recommendations.append("✗ Checkpoint loading may have failed")
        recommendations.append("  → Verify checkpoint file isn't corrupted")
        recommendations.append("  → Re-download pretrained weights")
        recommendations.append("  → Check model architecture matches checkpoint")
    
    if not recommendations:
        recommendations.append("✓ No obvious issues detected")
        recommendations.append("  → Loss spike might be random fluctuation")
        recommendations.append("  → Continue training and monitor")
    
    # Compile final report
    report = {
        "status": overall_status,
        "likely_causes": likely_causes if likely_causes else ["No obvious cause detected"],
        "recommendations": recommendations,
        "cause_scores": cause_scores,
        "findings": {
            "loss": loss_check,
            "gradients": grad_check,
            "learning_rate": lr_check,
            "batch": batch_check,
            "activations": activation_check,
        }
    }
    
    print("\n" + "="*70)
    print("DIAGNOSIS SUMMARY")
    print("="*70)
    print(f"\nOverall Status: {overall_status.upper()}")
    print(f"\nLikely Causes:")
    for cause in report["likely_causes"]:
        print(f"  • {cause}")
    print(f"\nRecommendations:")
    for rec in report["recommendations"]:
        print(f"  {rec}")
    print("="*70 + "\n")
    
    return report


def plot_loss_diagnostics(losses: List[float], title: str = "Training Loss"):
    """Plot loss curve with spike detection.
    
    Args:
        losses: List of loss values
        title: Plot title
    """
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        
        losses_arr = np.array(losses)
        steps = np.arange(len(losses_arr))
        
        # Raw loss
        axes[0].plot(steps, losses_arr, alpha=0.5, label="Raw loss")
        axes[0].set_title(title)
        axes[0].set_ylabel("Loss")
        axes[0].grid(True, alpha=0.3)
        
        # Moving average
        window = min(100, len(losses_arr) // 10)
        if window > 1:
            moving_avg = np.convolve(losses_arr, np.ones(window)/window, mode='valid')
            axes[0].plot(np.arange(len(moving_avg)) + window//2, moving_avg, 
                        label=f"Moving avg (window={window})", linewidth=2)
        
        axes[0].legend()
        
        # Log scale
        axes[1].semilogy(steps, np.maximum(losses_arr, 1e-8), alpha=0.5)
        axes[1].set_xlabel("Step")
        axes[1].set_ylabel("Loss (log scale)")
        axes[1].grid(True, alpha=0.3, which="both")
        
        plt.tight_layout()
        plt.savefig('loss_diagnostics.png', dpi=150, bbox_inches='tight')
        print("✓ Saved loss_diagnostics.png")
        plt.show()
    
    except ImportError:
        print("⚠ Matplotlib not available - skipping plot")