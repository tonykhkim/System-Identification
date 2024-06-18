function inputplotfunction(name,input,time)
    figure('Name',name);
    plot(time,input)    
    title('Steering wheel angle[rad]')
    xlabel('Time[s]')
    ylabel('δ [rad]')
end