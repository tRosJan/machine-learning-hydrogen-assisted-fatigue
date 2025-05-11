function fatigue = hafcgr(steel_grade, x, y)
    % Input:
    % steel_grade - The grade of the steel: 52, 70, or 100
    % x, y - Input for K and P
    % Output:
    % fatigue - The calculated fatigue e.g hafcgr(70, 15, 5)

    % Define coefficients for X52, X70, and X100
    coefficients = struct;

    % Coefficients for X52
    coefficients.X52 = [10.32, -5.325, 0.5227, 0.5694, 0.0752, -0.09318, -0.02495, ...
                        -0.005742, 0.001577, 0.003013, 0.0004034, 6.273e-05, 9.97e-05, -0.0001209];

    % Coefficients for X70
    coefficients.X70 = [-37.94, 6.592, 0.6737, -0.4664, -0.09507, -0.005882, 0.01464, ...
                        0.004374, 0.0005436, 2.146e-05, -0.0001706, -6.635e-05, -1.168e-05, -1.101e-06];

    % Coefficients for X100
    coefficients.X100 = [-8.842, 0.3591, 0.1054, 0.01571, -0.01063, 0.003392, -0.001472, ...
                         0.0005221, -0.000431, 5.916e-05, 2.656e-05, -9.85e-06, 1.354e-05, -4.381e-06];

    % Choose appropriate set of coefficients based on steel grade input
    switch steel_grade
        case 52
            coeff = coefficients.X52;
        case 70
            coeff = coefficients.X70;
        case 100
            coeff = coefficients.X100;
        otherwise
            error('Invalid steel grade. Please enter 52, 70, or 100.');
    end

    % Calculate the fatigue using the polynomial model
    fatigue = coeff(1) + coeff(2)*x + coeff(3)*y + coeff(4)*x^2 + coeff(5)*x*y + coeff(6)*y^2 + ...
              coeff(7)*x^3 + coeff(8)*x^2*y + coeff(9)*x*y^2 + coeff(10)*y^3 + coeff(11)*x^4 + ...
              coeff(12)*x^3*y + coeff(13)*x^2*y^2 + coeff(14)*x*y^3;
    fatigue = 10^fatigue;
end
