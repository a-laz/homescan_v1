# TODO List

## High Priority Features

### 3D Bin Packing for Truck Loading
- [ ] Implement 3D bin packing algorithm for efficient truck loading
  - [ ] Create truck dimensions model/input
  - [ ] Develop packing algorithm considering:
    - Item dimensions
    - Item fragility
    - Loading/unloading order
    - Weight distribution
  - [ ] Visualize packing solution in 3D
  - [ ] Generate loading/unloading instructions

#### Technical Considerations
- Algorithm selection (genetic algorithm, simulated annealing, etc.)
- Performance optimization for real-time calculations
- Handling irregular shapes and soft items
- Weight limits and distribution

#### User Interface
- [ ] Add truck selection/configuration
- [ ] Show packing visualization
- [ ] Provide step-by-step loading instructions
- [ ] Allow manual adjustments to packing solution

## Future Enhancements
- Support for multiple trucks
- Custom item constraints (must be upright, can't be stacked, etc.)
- Load balancing optimization
- Integration with route planning