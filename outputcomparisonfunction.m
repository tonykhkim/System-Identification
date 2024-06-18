function [error1, error2, rmse1, rmse2]=outputcomparisonfunction(time,x_dot1,x1_k,x3_k,comparison_index)

    comparison_range = comparison_index:(comparison_index+99);
    
    figure('Name','output comparison')
    subplot(2,2,1)
    plot(time(comparison_range),x_dot1(comparison_range,1),'g')
    hold on
    plot(time(comparison_range),x1_k(comparison_range,1),'r')
    plot(time(comparison_range),x3_k(comparison_range,1),'b')
    grid on
    title('Lateral Position','FontSize',15)
    legend('Input Data(고속주회로)','고속주회로 Model','K-city Model')
    xlabel('Time[s]','FontSize',10)
    ylabel('y [m]','FontSize',10)
    
    subplot(2,2,2)
    plot(time(comparison_range),x_dot1(comparison_range,2),'g')
    hold on
    plot(time(comparison_range),x1_k(comparison_range,2),'r')
    plot(time(comparison_range),x3_k(comparison_range,2),'b')
    grid on
    title('Lateral Velocity','FontSize',15)
    legend('Input Data(고속주회로)','고속주회로 Model','K-city Model')
    xlabel('Time[s]','FontSize',10)
    ylabel('dy/dt [m/s]','FontSize',10)

    subplot(2,2,3)
    plot(time(comparison_range),x_dot1(comparison_range,3),'g')
    hold on
    plot(time(comparison_range),x1_k(comparison_range,3),'r')
    plot(time(comparison_range),x3_k(comparison_range,3),'b')
    grid on
    title('Yaw','FontSize',15)
    legend('Input Data(고속주회로)','고속주회로 Model','K-city Model')
    xlabel('Time[s]','FontSize',10)
    ylabel('ψ [rad]','FontSize',10)

    subplot(2,2,4)
    plot(time(comparison_range),x_dot1(comparison_range,4),'g')
    hold on
    plot(time(comparison_range),x1_k(comparison_range,4),'r')
    plot(time(comparison_range),x3_k(comparison_range,4),'b')
    grid on
    title('Yaw Rate','FontSize',15)
    legend('Input Data(고속주회로)','고속주회로 Model','K-city Model')
    xlabel('Time[s]','FontSize',10)
    ylabel('dψ/dt [rad/s]','FontSize',10)

    error_xdot1_yn1_1 = x_dot1(comparison_range,1)-x1_k(comparison_range,1);
    error_xdot1_yn1_2 = x_dot1(comparison_range,2)-x1_k(comparison_range,2);
    error_xdot1_yn1_3 = x_dot1(comparison_range,3)-x1_k(comparison_range,3);
    error_xdot1_yn1_4 = x_dot1(comparison_range,4)-x1_k(comparison_range,4);
    error1 = [error_xdot1_yn1_1 error_xdot1_yn1_2 error_xdot1_yn1_3 error_xdot1_yn1_4];
    rmse1_1 = sqrt(mean((error_xdot1_yn1_1).^2));
    rmse1_2 = sqrt(mean((error_xdot1_yn1_2).^2));
    rmse1_3 = sqrt(mean((error_xdot1_yn1_3).^2));
    rmse1_4 = sqrt(mean((error_xdot1_yn1_4).^2));
    rmse1 = [rmse1_1 rmse1_2 rmse1_3 rmse1_4];
    
    error_xdot1_yn3_1 = x_dot1(comparison_range,1)-x3_k(comparison_range,1);
    error_xdot1_yn3_2 = x_dot1(comparison_range,2)-x3_k(comparison_range,2);
    error_xdot1_yn3_3 = x_dot1(comparison_range,3)-x3_k(comparison_range,3);
    error_xdot1_yn3_4 = x_dot1(comparison_range,4)-x3_k(comparison_range,4);
    error2 = [error_xdot1_yn3_1 error_xdot1_yn3_2 error_xdot1_yn3_3 error_xdot1_yn3_4];
    rmse2_1 = sqrt(mean((error_xdot1_yn3_1).^2));
    rmse2_2 = sqrt(mean((error_xdot1_yn3_2).^2));
    rmse2_3 = sqrt(mean((error_xdot1_yn3_3).^2));
    rmse2_4 = sqrt(mean((error_xdot1_yn3_4).^2));
    rmse2 = [rmse2_1 rmse2_2 rmse2_3 rmse2_4];

end