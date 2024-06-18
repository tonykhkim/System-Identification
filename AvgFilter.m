function avg = AvgFilter(x)

    persistent prevAvg1
    persistent prevAvg2
    persistent prevAvg3
    persistent prevAvg4
    persistent kk 
    persistent firstRun

    if isempty(firstRun)
        kk=1;
        prevAvg1 = 0;
        prevAvg2 = 0;
        prevAvg3 = 0;
        prevAvg4 = 0;
        firstRun = 1;

    end

    alpha = (kk-1) / kk;

    avg1 = alpha * prevAvg1 + (1-alpha)*x(:,1);
    avg2 = alpha * prevAvg2 + (1-alpha)*x(:,2);
    avg3 = alpha * prevAvg3 + (1-alpha)*x(:,3);
    avg4 = alpha * prevAvg4 + (1-alpha)*x(:,4);
    
    prevAvg1 = avg1;
    prevAvg2 = avg2;
    prevAvg3 = avg3;
    prevAvg4 = avg4;
    
    avg = [avg1 avg2 avg3 avg4];

    kk = kk+1;

end
