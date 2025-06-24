### A python program/class to handle difficult fit models and analysis of pump-probe spectroscopy data (WARNING: WIP and by no means functional!!!)

### Absoluteley not correct, just placeholder .md:
## Features

- Flexible fitting of complex models to pump-probe spectroscopy data
- Support for custom model definitions
- Automated parameter estimation and error analysis
- Visualization tools for data and fit results
- Batch processing for multiple datasets

## Installation

```bash
pip install better-fits
```

## Usage

```python
from better_fits import FitModel

# Load your data
data = load_data('your_data.csv')

# Define your model
def custom_model(x, a, b, c):
    return a * np.exp(-b * x) + c

# Initialize and fit
fit = FitModel(data, model=custom_model)
fit.run()
fit.plot()
```

## Example

See the [examples](examples/) directory for sample scripts and datasets.

## Requirements

- Python 3.7+
- numpy
- scipy
- matplotlib

## Contributing

Contributions are welcome! Please open issues or submit pull requests.

## License

This project is licensed under the MIT License.