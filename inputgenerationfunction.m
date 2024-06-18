function input_data_rad = inputgenerationfunction(x)
    gt_bag = rosbag(x);
    steering = select(gt_bag,'Topic','/groot/chassis_can/platform_standard');
    msgStructs4 = readMessages(steering,'DataFormat','struct');
    msgStructs4{1};
    input_data_deg = cellfun(@(m) double(m.WsaDeg),msgStructs4);
%     input_data = cellfun(@(m) double(m.SwaDeg),msgStructs4);
    input_data_rad=deg2rad(input_data_deg);
    
    %%%%%%%%%%%%inverse data generation%%%%%%%%%%%%%%%
%     input_data_inverse=-1*input_data;


    

end

