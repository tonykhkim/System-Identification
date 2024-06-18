function [std1, std2] = normaldistributionfunction(error1,error2)
    
%     search_range = search_index:(search_index+99);
%     error_xdot1_yn1_1 = x_dot1(search_range,1)-x1_k(search_range,1);
%     error_xdot1_yn1_2 = x_dot1(search_range,2)-x1_k(search_range,2);
%     error_xdot1_yn1_3 = x_dot1(search_range,3)-x1_k(search_range,3);
%     error_xdot1_yn1_4 = x_dot1(search_range,4)-x1_k(search_range,4);
%     error1 = [error_xdot1_yn1_1 error_xdot1_yn1_2 error_xdot1_yn1_3 error_xdot1_yn1_4];
%     
%     error_xdot1_yn3_1 = x_dot1(search_range,1)-x3_k(search_range,1);
%     error_xdot1_yn3_2 = x_dot1(search_range,2)-x3_k(search_range,2);
%     error_xdot1_yn3_3 = x_dot1(search_range,3)-x3_k(search_range,3);
%     error_xdot1_yn3_4 = x_dot1(search_range,4)-x3_k(search_range,4);
%     error2 = [error_xdot1_yn3_1 error_xdot1_yn3_2 error_xdot1_yn3_3 error_xdot1_yn3_4];
        
    figure('Name','Normal distribution 고속주회로 Model & K-city Model in search range');
    subplot(2,2,1);
    [V1_1,M1_1] = var(error1(:,1));
    std1_1 = sqrt(V1_1);
    normal_distribution1_1 = makedist('Normal', 'mu', M1_1, 'sigma', std1_1);
    x1_1 = linspace(M1_1 - 3 * std1_1, M1_1 + 3 * std1_1, 1000); % 플롯을 위한 x 값 범위
    y1_1 = pdf(normal_distribution1_1, x1_1); % 확률 밀도 함수 계산
    
    [V2_1,M2_1] = var(error2(:,1));
    std2_1 = sqrt(V2_1);
    normal_distribution2_1 = makedist('Normal', 'mu', M2_1, 'sigma', std2_1);
    x2_1 = linspace(M2_1 - 3 * std2_1, M2_1 + 3 * std2_1, 1000); % 플롯을 위한 x 값 범위
    y2_1 = pdf(normal_distribution2_1, x2_1); % 확률 밀도 함수 계산
    
    plot(x1_1, y1_1,'r-', 'LineWidth', 1.5);
    hold on;
    plot(x2_1, y2_1,'b-', 'LineWidth', 1.5);
    xline(M1_1, 'r--');
    xline(M2_1, 'b--');
    legend("고속주회로 Model","K-city Model");
    title('Normal distribution of Lateral position error','FontSize',15)
    xlabel('error');
    ylabel('probability density');
    grid on;
    
    subplot(2,2,2);
    [V1_2,M1_2] = var(error1(:,2));
    std1_2 = sqrt(V1_2);
    normal_distribution1_2 = makedist('Normal', 'mu', M1_2, 'sigma', std1_2);
    x1_2 = linspace(M1_2 - 3 * std1_2, M1_2 + 3 * std1_2, 1000); % 플롯을 위한 x 값 범위
    y1_2 = pdf(normal_distribution1_2, x1_2); % 확률 밀도 함수 계산
    
    [V2_2,M2_2] = var(error2(:,2));
    std2_2 = sqrt(V2_2);
    normal_distribution2_2 = makedist('Normal', 'mu', M2_2, 'sigma', std2_2);
    x2_2 = linspace(M2_2 - 3 * std2_2, M2_2 + 3 * std2_2, 1000); % 플롯을 위한 x 값 범위
    y2_2 = pdf(normal_distribution2_2, x2_2); % 확률 밀도 함수 계산
    
    plot(x1_2, y1_2,'r-', 'LineWidth', 1.5);
    hold on;
    plot(x2_2, y2_2,'b-', 'LineWidth', 1.5);
    xline(M1_2, 'r--');
    xline(M2_2, 'b--');
    legend("고속주회로 Model","K-city Model");
    title('Normal distribution of Lateral velocity error','FontSize',15)
    xlabel('error');
    ylabel('probability density');
    grid on;
    
    subplot(2,2,3);
    [V1_3,M1_3] = var(error1(:,3));
    std1_3 = sqrt(V1_3);
    normal_distribution1_3 = makedist('Normal', 'mu', M1_3, 'sigma', std1_3);
    x1_3 = linspace(M1_3 - 3 * std1_3, M1_3 + 3 * std1_3, 1000); % 플롯을 위한 x 값 범위
    y1_3 = pdf(normal_distribution1_3, x1_3); % 확률 밀도 함수 계산
    
    [V2_3,M2_3] = var(error2(:,3));
    std2_3 = sqrt(V2_3);
    normal_distribution2_3 = makedist('Normal', 'mu', M2_3, 'sigma', std2_3);
    x2_3 = linspace(M2_3 - 3 * std2_3, M2_3 + 3 * std2_3, 1000); % 플롯을 위한 x 값 범위
    y2_3 = pdf(normal_distribution2_3, x2_3); % 확률 밀도 함수 계산
    
    plot(x1_3, y1_3,'r-', 'LineWidth', 1.5);
    hold on;
    plot(x2_3, y2_3,'b-', 'LineWidth', 1.5);
    xline(M1_3, 'r--');
    xline(M2_3, 'b--');
    legend("고속주회로 Model","K-city Model");
    title('Normal distribution of yaw error','FontSize',15)
    xlabel('error');
    ylabel('probability density');
    grid on;
    
    subplot(2,2,4);
    [V1_4,M1_4] = var(error1(:,4));
    std1_4 = sqrt(V1_4);
    normal_distribution1_4 = makedist('Normal', 'mu', M1_4, 'sigma', std1_4);
    x1_4 = linspace(M1_4 - 3 * std1_4, M1_4 + 3 * std1_4, 1000); % 플롯을 위한 x 값 범위
    y1_4 = pdf(normal_distribution1_4, x1_4); % 확률 밀도 함수 계산
    
    [V2_4,M2_4] = var(error2(:,4));
    std2_4 = sqrt(V2_4);
    normal_distribution2_4 = makedist('Normal', 'mu', M2_4, 'sigma', std2_4);
    x2_4 = linspace(M2_4 - 3 * std2_4, M2_4 + 3 * std2_4, 1000); % 플롯을 위한 x 값 범위
    y2_4 = pdf(normal_distribution2_4, x2_4); % 확률 밀도 함수 계산
    
    plot(x1_4, y1_4,'r-', 'LineWidth', 1.5);
    hold on;
    plot(x2_4, y2_4,'b-', 'LineWidth', 1.5);
    xline(M1_4, 'r--');
    xline(M2_4, 'b--');
    legend("고속주회로 Model","K-city Model");
    title('Normal distribution of yaw angle rate error','FontSize',15)
    xlabel('error');
    ylabel('probability density');
    grid on;

    std1=[std1_1 std1_2 std1_3 std1_4];
    std2=[std2_1 std2_2 std2_3 std2_4];

end