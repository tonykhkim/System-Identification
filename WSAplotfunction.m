function WSAplotfunction(WSA_rad,time,name)

    figure('Name',name);
    plot(time,WSA_rad)
    hold on
    title('WSA_{rad}')
    xlabel('Time[s]')
    ylabel('δ [rad]')

end

