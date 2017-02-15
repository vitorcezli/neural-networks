% This code uses the ANN stochastic gradient descent to minimize the cost of the
% forward propagation.
% 
% The function invocation is:
%  
% [ WH, WO, C ] = ann_stochastic_training( TI, nh, TO, ne, lr )
%  
% WH -> weights used to calculate the hidden units' activation values
% WO -> weights used to calculate the output units' activation values
% C -> a vector, representing the cost on each epoch
% TI -> a matrix with the inputs
% nh -> number of hidden units (bias unit is not counted)
% TO -> a matrix with the outputs
% ne -> number of epochs
% lr -> the learning rate

function [ wtoh, wtoo, cost ] = ann_stochastic_training( training_inputs, ...
    number_hidden, training_outputs, number_epochs, learning_rate )

    % randomize the weights' initial values
    wtoh = ( randi( 99, [ number_hidden, size( training_inputs, 2 ) + 1 ] ) ...
        - 50 ) ./ 1000;
    wtoo = ( randi( 99, [ size( training_outputs, 2 ), number_hidden + 1 ] ) ...
        - 50 ) ./ 1000;
    
    for j = 1 : number_epochs
    
        cost( end + 1, 1 ) = 0;
        
        for i = 1 : size( training_inputs, 1 )
            % forward propagation
            inputs = [ 1; training_inputs( i, : )' ];
            hidden_units = sigmoid( wtoh * inputs );
            hidden_units = [ 1; hidden_units ];
            outputs = sigmoid( wtoo * hidden_units );
            
            % adds the error to the cost
            cost( end, 1 ) += sum( ( training_outputs( i, : )' - outputs ) .^ 2 );
            
            % calculate the outputs' error
            errors = ( outputs .* ( 1 - outputs ) ) .* ...
                ( training_outputs( i, : )' - outputs );
            
            % backpropagation
            hidden_errors = ( hidden_units .* ( 1 - hidden_units ) ) .* ...
                ( wtoo' * errors );
            hidden_errors = hidden_errors( 2 : end, : );

            % atualize the weights
            wtoh = wtoh + learning_rate * ( hidden_errors * inputs' );
            wtoo = wtoo + learning_rate * ( errors * hidden_units' );
        endfor
        
        cost( end, 1 ) = cost( end, 1 ) / 2;
        
    endfor
    
endfunction