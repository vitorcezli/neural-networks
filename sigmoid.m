function sig = sigmoid( inputs )

    inputs = inputs( : );
    sig = 1 ./ ( 1 + e .^ ( -inputs ) );

endfunction