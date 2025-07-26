import matplotlib.pyplot as plt

# Data definitions
cifar_sym_noise = [0, 20, 40, 60]
cifar_sym = {
    'NNK weights': [0.979, 0.963, 0.953, 0.953],
    'NNK ensemble': [0.978, 0.963, 0.954, 0.954],
    'NNK diam ratio': [0.980, 0.966, 0.956, 0.956],
    'NNK diam ratio ens': [0.976, 0.962, 0.941, 0.941]
}

cifar_asym_noise = [20, 30, 40]
cifar_asym = {
    'NNK weights': [0.956, 0.923, 0.840],
    'NNK ensemble': [0.956, 0.924, 0.843],
    'NNK diam ratio': [0.940, 0.896, 0.821],
    'NNK diam ratio ens': [0.938, 0.894, 0.818]
}

derma_sym_noise = [0, 20, 40, 60]
derma_sym = {
    'NNK weights': [0.735, 0.732, 0.715, 0.614],
    'NNK ensemble': [0.734, 0.730, 0.716, 0.622],
    'NNK diam ratio': [0.759, 0.734, 0.676, 0.498],
    'NNK diam ratio ens': [0.759, 0.732, 0.670, 0.492]
}

derma_asym_noise = [20, 30, 40]
derma_asym = {
    'NNK weights': [0.699, 0.721, 0.600],
    'NNK ensemble': [0.699, 0.722, 0.602],
    'NNK diam ratio': [0.705, 0.717, 0.548],
    'NNK diam ratio ens': [0.703, 0.715, 0.546]
}

markers = ['o', 's', '^', 'd']

def plot_nnk(noise, data, title, filename):
    plt.figure()
    for (method, vals), m in zip(data.items(), markers):
        plt.plot(noise, vals, linestyle='-', marker=m, color='C0', label=method)
    plt.xlabel('Noise Ratio (%)')
    plt.ylabel('Accuracy')
    plt.title(title)
    plt.ylim(0,1)
    plt.legend()
    plt.grid(True)
    plt.savefig(f'./vis/{filename}', bbox_inches='tight')
    plt.close()

# Generate and save plots
plot_nnk(cifar_sym_noise, cifar_sym, 'CIFAR-10 Symmetric NNK Methods', 'cifar_sym_nnk.png')
plot_nnk(cifar_asym_noise, cifar_asym, 'CIFAR-10 Asymmetric NNK Methods', 'cifar_asym_nnk.png')
plot_nnk(derma_sym_noise, derma_sym, 'DermaMNIST Symmetric NNK Methods', 'derma_sym_nnk.png')
plot_nnk(derma_asym_noise, derma_asym, 'DermaMNIST Asymmetric NNK Methods', 'derma_asym_nnk.png')
