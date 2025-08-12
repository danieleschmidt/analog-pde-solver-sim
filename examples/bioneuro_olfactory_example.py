#!/usr/bin/env python3
"""
Bio-Neuromorphic Olfactory Fusion Example

Demonstrates the advanced bio-inspired analog computing capabilities
for chemical gradient detection and processing using olfactory-inspired
neural networks combined with analog PDE solving.
"""

import numpy as np
import matplotlib.pyplot as plt
from analog_pde_solver.research.bioneuro_olfactory_fusion import (
    BioneuroOlfactoryFusion,
    OlfactoryReceptorConfig,
    MitralCellNetwork,
    benchmark_bioneuro_olfactory_performance
)


def create_complex_chemical_environment():
    """Create a complex multi-source chemical environment for testing."""
    size = 64
    x, y = np.meshgrid(np.linspace(0, 20, size), np.linspace(0, 20, size))
    
    # Primary chemical plume (e.g., food source)
    primary_source = 3.0 * np.exp(-((x-15)**2 + (y-10)**2) / 8)
    
    # Secondary plume (e.g., predator pheromone)  
    secondary_source = 2.0 * np.exp(-((x-5)**2 + (y-15)**2) / 6)
    
    # Background chemical gradient
    background = 0.5 * np.exp(-((x-10)**2 + (y-5)**2) / 20)
    
    # Add noise to simulate real-world conditions
    noise = np.random.normal(0, 0.1, (size, size))
    
    return {
        'primary': np.clip(primary_source + noise, 0, None),
        'secondary': np.clip(secondary_source + noise, 0, None),
        'background': np.clip(background + noise, 0, None),
        'combined': np.clip(primary_source + secondary_source + background + noise, 0, None)
    }


def demonstrate_olfactory_processing():
    """Demonstrate bio-neuromorphic olfactory processing capabilities."""
    print("🧠 Bio-Neuromorphic Olfactory Processing Demo")
    print("=" * 50)
    
    # Create advanced olfactory system
    print("Initializing bio-neuromorphic olfactory system...")
    fusion_engine = BioneuroOlfactoryFusion(
        receptor_config=OlfactoryReceptorConfig(
            num_receptors=1024,  # High-density receptor array
            sensitivity_range=(1e-15, 1e-4),
            response_time=0.0005,
            adaptation_rate=0.05,
            noise_level=0.01
        ),
        mitral_config=MitralCellNetwork(
            num_cells=256,
            inhibition_radius=4.0,
            inhibition_strength=0.6,
            temporal_dynamics=True,
            oscillation_frequency=45.0
        ),
        crossbar_size=256
    )
    
    # Create complex chemical environment
    print("Creating complex chemical environment...")
    environment = create_complex_chemical_environment()
    
    # Process each chemical signal
    print("Processing chemical signals through olfactory pathway...")
    results = {}
    
    for signal_name, signal_data in environment.items():
        print(f"  Processing {signal_name} signal...")
        result = fusion_engine.detect_chemical_gradients(signal_data)
        results[signal_name] = result
        
        print(f"    Gradient magnitude: {np.mean(result['gradient_magnitude']):.6f}")
        print(f"    Max gradient: {np.max(result['gradient_magnitude']):.6f}")
    
    # Multi-modal signal fusion
    print("\nPerforming multi-modal signal fusion...")
    fused_gradients = fusion_engine.fuse_multimodal_signals(
        [environment['primary'], environment['secondary'], environment['background']],
        signal_weights=[0.6, 0.3, 0.1]  # Prioritize primary source
    )
    
    print(f"Fused signal strength: {np.mean(fused_gradients):.6f}")
    print(f"Max fused signal: {np.max(fused_gradients):.6f}")
    
    # Environmental adaptation
    print("\nAdapting to environmental statistics...")
    adaptation_signals = [
        environment['combined'] + np.random.normal(0, 0.05, environment['combined'].shape)
        for _ in range(5)  # Multiple environment samples
    ]
    
    fusion_engine.adapt_to_environment(adaptation_signals, learning_rate=0.02)
    
    # Get comprehensive metrics
    metrics = fusion_engine.get_processing_metrics()
    
    print("\n📊 Bio-Neuromorphic System Metrics:")
    print(f"  Receptor Array: {metrics['receptor_metrics']['num_receptors']} receptors")
    print(f"  Mitral Network: {metrics['mitral_metrics']['num_cells']} cells")
    print(f"  Adaptation Level: {metrics['receptor_metrics']['adaptation_state']['mean']:.3f}")
    print(f"  Neural Activity: {metrics['mitral_metrics']['current_state']['active_fraction']:.3f}")
    print(f"  Inhibition Strength: {metrics['mitral_metrics']['inhibition_strength']:.3f}")
    print(f"  Convergence Ratio: {metrics['glomerular_metrics']['convergence_ratio']:.1f}")
    
    return {
        'fusion_engine': fusion_engine,
        'environment': environment,
        'results': results,
        'fused_gradients': fused_gradients,
        'metrics': metrics
    }


def visualize_olfactory_processing(demo_results):
    """Create comprehensive visualization of olfactory processing."""
    print("\n📈 Generating olfactory processing visualizations...")
    
    environment = demo_results['environment']
    results = demo_results['results']
    fused_gradients = demo_results['fused_gradients']
    
    # Create figure with subplots
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    fig.suptitle('Bio-Neuromorphic Olfactory Processing Pipeline', fontsize=16)
    
    # Row 1: Original chemical fields
    im1 = axes[0,0].imshow(environment['primary'], cmap='Reds', aspect='auto')
    axes[0,0].set_title('Primary Chemical Source')
    axes[0,0].set_xlabel('Spatial X')
    axes[0,0].set_ylabel('Spatial Y')
    plt.colorbar(im1, ax=axes[0,0])
    
    im2 = axes[0,1].imshow(environment['secondary'], cmap='Blues', aspect='auto')
    axes[0,1].set_title('Secondary Chemical Source')
    axes[0,1].set_xlabel('Spatial X')
    plt.colorbar(im2, ax=axes[0,1])
    
    im3 = axes[0,2].imshow(environment['background'], cmap='Greens', aspect='auto')
    axes[0,2].set_title('Background Chemical Field')
    axes[0,2].set_xlabel('Spatial X')
    plt.colorbar(im3, ax=axes[0,2])
    
    im4 = axes[0,3].imshow(environment['combined'], cmap='viridis', aspect='auto')
    axes[0,3].set_title('Combined Chemical Environment')
    axes[0,3].set_xlabel('Spatial X')
    plt.colorbar(im4, ax=axes[0,3])
    
    # Row 2: Gradient magnitudes
    for i, (name, result) in enumerate(list(results.items())[:4]):
        im = axes[1,i].imshow(result['gradient_magnitude'], cmap='plasma', aspect='auto')
        axes[1,i].set_title(f'{name.capitalize()} Gradients')
        axes[1,i].set_xlabel('Spatial X')
        if i == 0:
            axes[1,i].set_ylabel('Spatial Y')
        plt.colorbar(im, ax=axes[1,i])
    
    # Row 3: Processing stages and fusion
    # Receptor response (primary signal)
    receptor_response = results['primary']['receptor_response']
    im5 = axes[2,0].imshow(receptor_response, cmap='hot', aspect='auto')
    axes[2,0].set_title('Olfactory Receptor Response')
    axes[2,0].set_xlabel('Spatial X')
    axes[2,0].set_ylabel('Spatial Y')
    plt.colorbar(im5, ax=axes[2,0])
    
    # Mitral cell output
    mitral_output = results['primary']['mitral_output']
    im6 = axes[2,1].imshow(mitral_output, cmap='coolwarm', aspect='auto')
    axes[2,1].set_title('Mitral Cell Processing')
    axes[2,1].set_xlabel('Spatial X')
    plt.colorbar(im6, ax=axes[2,1])
    
    # Gradient direction
    gradient_direction = results['combined']['gradient_direction']
    im7 = axes[2,2].imshow(gradient_direction, cmap='hsv', aspect='auto')
    axes[2,2].set_title('Gradient Direction')
    axes[2,2].set_xlabel('Spatial X')
    plt.colorbar(im7, ax=axes[2,2])
    
    # Fused signal
    im8 = axes[2,3].imshow(fused_gradients, cmap='magma', aspect='auto')
    axes[2,3].set_title('Multi-Modal Fused Signal')
    axes[2,3].set_xlabel('Spatial X')
    plt.colorbar(im8, ax=axes[2,3])
    
    plt.tight_layout()
    plt.savefig('bioneuro_olfactory_processing.png', dpi=300, bbox_inches='tight')
    print("  Saved visualization: bioneuro_olfactory_processing.png")
    
    # Performance metrics plot
    metrics = demo_results['metrics']
    
    fig2, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    fig2.suptitle('Bio-Neuromorphic System Performance Metrics', fontsize=14)
    
    # Receptor adaptation histogram
    adaptation_data = metrics['receptor_metrics']['adaptation_state']
    ax1.hist(demo_results['fusion_engine'].receptor_adaptation, bins=50, alpha=0.7, color='blue')
    ax1.axvline(adaptation_data['mean'], color='red', linestyle='--', label=f"Mean: {adaptation_data['mean']:.3f}")
    ax1.set_xlabel('Adaptation Level')
    ax1.set_ylabel('Number of Receptors')
    ax1.set_title('Receptor Adaptation Distribution')
    ax1.legend()
    
    # Mitral cell state
    mitral_data = metrics['mitral_metrics']['current_state']
    ax2.hist(demo_results['fusion_engine'].mitral_state, bins=30, alpha=0.7, color='green')
    ax2.axvline(mitral_data['mean'], color='red', linestyle='--', label=f"Mean: {mitral_data['mean']:.3f}")
    ax2.set_xlabel('Mitral Cell Activity')
    ax2.set_ylabel('Number of Cells')
    ax2.set_title('Mitral Cell Activity Distribution')
    ax2.legend()
    
    # Processing performance comparison
    signal_names = list(results.keys())
    gradient_strengths = [np.mean(results[name]['gradient_magnitude']) for name in signal_names]
    
    bars = ax3.bar(signal_names, gradient_strengths, color=['red', 'blue', 'green', 'purple'])
    ax3.set_ylabel('Mean Gradient Magnitude')
    ax3.set_title('Gradient Detection Performance')
    ax3.tick_params(axis='x', rotation=45)
    
    # Network connectivity visualization (sample)
    convergence_matrix = demo_results['fusion_engine'].glomerular_map[:16, :64]  # Sample
    im = ax4.imshow(convergence_matrix, cmap='viridis', aspect='auto')
    ax4.set_xlabel('Receptor Index (sample)')
    ax4.set_ylabel('Mitral Cell Index (sample)')
    ax4.set_title('Glomerular Convergence Pattern')
    plt.colorbar(im, ax=ax4)
    
    plt.tight_layout()
    plt.savefig('bioneuro_metrics.png', dpi=300, bbox_inches='tight')
    print("  Saved metrics: bioneuro_metrics.png")
    
    plt.show()


def main():
    """Main function demonstrating bio-neuromorphic olfactory fusion."""
    print("🚀 Bio-Neuromorphic Olfactory Fusion Example")
    print("=" * 55)
    
    try:
        # Run comprehensive demonstration
        demo_results = demonstrate_olfactory_processing()
        
        # Generate visualizations
        visualize_olfactory_processing(demo_results)
        
        # Run research benchmark
        print("\n🔬 Running research-grade benchmark...")
        benchmark_results = benchmark_bioneuro_olfactory_performance()
        
        print("\n✅ Bio-neuromorphic olfactory fusion example completed successfully!")
        print("\nKey Innovations Demonstrated:")
        print("  🧠 Bio-inspired olfactory receptor arrays")
        print("  🔗 Glomerular convergence processing")
        print("  ⚡ Mitral cell lateral inhibition networks")
        print("  🧮 Analog PDE gradient computation")
        print("  🔄 Multi-modal signal fusion")
        print("  📊 Environmental adaptation")
        print("  📈 Comprehensive performance metrics")
        
        return {
            'demo_results': demo_results,
            'benchmark_results': benchmark_results
        }
        
    except Exception as e:
        print(f"❌ Error in bio-neuromorphic demonstration: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    results = main()