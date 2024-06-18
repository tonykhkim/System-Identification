function analFullmotion2(gt_data,input_data,minmax_input,dataname)
    %analLocalmotion함수에서 출력한 minmax_input을 입력받아 
    % 원본 steering input data figure 위에 
    % 입력 받은 minmax_input을 그리는 함수  

    gt_bag = rosbag(gt_data);
    gt_vel_x=select(gt_bag,"Time",[gt_bag.StartTime gt_bag.EndTime],"Topic","/gt/vel");
    ts_x=timeseries(gt_vel_x,'Vector.X');      %종방향 속도
    time_x=ts_x.Time(:)-ts_x.Time(1);
    
    figure('Name',[dataname ' yline original full data'])
    plot(time_x,input_data,'r')
    grid on
    title('Wheel Steering Angle','FontSize',15)
    xlabel('Time[s]','FontSize',10)
    ylabel('δ [rad]','FontSize',10)
    hold on
    yline(minmax_input(2),'g')
    yline(minmax_input(1),'b')
    legend('K-city1의 steering','고속1의 max steering','고속1의 min steering')

    figure('Name',[dataname ' zoom original full data'])
    plot(time_x,input_data,'r')
    grid on
    title('Wheel Steering Angle','FontSize',15)
    xlabel('Time[s]','FontSize',10)
    ylabel('δ [rad]','FontSize',10)
    hold on
    yline(minmax_input(2),'g')
    yline(minmax_input(1),'b')
    ylim([minmax_input(1)-0.01 minmax_input(2)+0.01])
    legend('K-city1의 steering','고속1의 max steering','고속1의 min steering')
end