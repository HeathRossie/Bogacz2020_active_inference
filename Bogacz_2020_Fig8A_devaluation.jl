# Julia code to generate Figure 8A of Bogacz (2020)
# Codes adapted from Bogacz (2020)'s Matlab implementation
# written by Hiroshi Matsui, PhD 
# 28 Feb 2021

using Distributions, Random, Plots
Random.seed!(123)


#------------------------------------------------------------------------------------------------------------------------------------------------#
# define functions
function extinction(train_trials, devaluation, BIN)
    # INPUTS
    # train_trials : the  number of  training trials 120 or 360 accoding to Dickinson's study on habit
    # devaluation : 0/1 to devalue or not 
    # BIN : bins of aggregation of trials, set to be 30 in the simulation

    dt = 0.001;          # step of the Euler method of numerical integration
    POSTEXT = 180.0;       #  post extinction trials = 180 trials
    TRIALS = Int(train_trials + POSTEXT);        #number of simulated trials
    CSTIME = 1.0;          # Time within a trial at which CS is present
    RTIME = 2.0;           # Time of the reward
    MICROT = 0.2;        # Duration of a micro-state
    ITER = 10;           # iteration of simulation
    alphag = 0.05;           # learning rate for the goal-directed
    alphah = 0.02;           # learning rate for the habit    
    action = zeros(Int(TRIALS));  # an action vector

    # result vector
    abin = zeros(ITER, Int(POSTEXT/BIN));
    
    # generate_state function
    s, sval, rperiod, NMICRO, T = generate_state(CSTIME, RTIME, MICROT, dt);

    # main loop of extinction 
    for i = 1:ITER
        
        # initialize_model function
        w, q, h, varianceg, varianceh = initialize_model(NMICRO);    

        # each truak loop
        for trial = 1:TRIALS

            # using simulate_trial function to simulate each trial

            # during training sessions
            if trial <= train_trials
                w_temp, q_temp, h_temp, varianceg_temp, varianceh_temp, not_used1, not_used2, not_used3, a = 
                    simulate_trial(w[trial,:], q[trial], h[trial], varianceg[trial], varianceh[trial], T/dt, MICROT, dt, s, sval, rperiod, rsat, alphag, alphah);
            
                
                # append results
                w = vcat(w, reshape(w_temp, 1, 5));
                q = hcat(q, q_temp);
                h = hcat(h, h_temp);
                varianceg = hcat(varianceg, varianceg_temp);
                varianceh = hcat(varianceh, varianceh_temp);
                
            else # then, during test trials (with or without devaluation)
                if (devaluation==1) & (trial == train_trials+1)
                    q[trial] = q[trial] * 0;
                    w[trial,:] = w[trial,:] .* 0;
                end

                w_temp, q_temp, h_temp, varianceg_temp, varianceh_temp, not_used1, not_used2, not_used3, a = 
                    simulate_trial(w[trial,:], q[trial], h[trial], varianceg[trial], varianceh[trial], T/dt, MICROT, dt, s, sval, rperiod, reff, alphag, alphah);
                
                # append results
                w = vcat(w, reshape(w_temp, 1, 5));
                q = hcat(q, q_temp);
                h = hcat(h, h_temp);
                varianceg = hcat(varianceg, varianceg_temp);
                varianceh = hcat(varianceh, varianceh_temp);
            end
            action[trial] = a[Int(RTIME/dt-1)];
        end

        
        ashape = reshape(action[train_trials+1:end], BIN, Int(round(POSTEXT/BIN)));
        abin[i,:] = mean(ashape, dims=1);
        print('.');
    end

    maction = mean(abin, dims=1);
    saction = std(abin, dims=1);

    return [maction, saction]
end


#------------------------------------------------------------------------------------------------------------------------------------------------#
# generate_state
function generate_state(CSTIME, RTIME, MICROT, dt)

    NMICRO = (RTIME - CSTIME) / MICROT;
    T = RTIME + MICROT; 
    
    s = zeros(Int(T/dt+1));
    s[Int(round(CSTIME/dt)):Int(round((RTIME+MICROT)/dt))] .= 1.0;
    sval = zeros(Int(NMICRO), Int(T/dt));

    for i = 1:NMICRO
        sval[Int(i), Int(round((CSTIME+(i-1)*MICROT)/dt)):Int(round((CSTIME+i*MICROT)/dt))] .= 1.0;
    end
    
    rperiod = zeros(Int(T/dt));
    rperiod[Int(round((RTIME)/dt)):Int(round((RTIME+MICROT)/dt))] .= 1;
    
    return s, sval, rperiod, NMICRO, T 
end


#------------------------------------------------------------------------------------------------------------------------------------------------#
function initialize_model(NSTATES)
    
    w = zeros(Int(NSTATES)) .+ 0.1;
    w = reshape(w, 1, length(w))
    q = [0.1];
    h = 0;
    varianceg = 1.0;
    varianceh = 100.0;

    return w, q, h, varianceg, varianceh
end




#------------------------------------------------------------------------------------------------------------------------------------------------#
# reward function 
function rsat(a, reward_noise=0.5)
    
    trueW = 5.0;          # maximum reward available
    trueQ = 3.0;          # true scalling of reward available
    
    r = trueW * tanh(a*trueQ/trueW) - a + rand(Normal(0, reward_noise), 1)[1];
    return r
end


function reff(a)
    r = -a;
    return r
end

#------------------------------------------------------------------------------------------------------------------------------------------------#
# main simulation function that simulates a trial
# and compute dynamics of each parameter, using Euler method


function simulate_trial(w, q, h, varianceg, varianceh, NTIME, MICROT, dt, s, sval, rperiod, rfun, alphag, alphah)
    
    
    # Learning rates
    alphacplus = 0.5;        #learning rate for the critic, when prediction error > 0
    alphacminus = 0.5;       #learning rate for the critic, when prediction error < 0
    alphaprecg = 0.05;       #learning rate for the variance
    alphaprech = 0.1;        #learning rate for the variance

    # Other parameters of the model
    lambda = 0.9;            #retention of eligibility trace over duration of a micro-state
    tau = 0.05;              #time constant of an reward
    taudelta = 0.02;         #time constant of an prediction error
    action_noise = 1;        #standard deviation of noise added to action 
    minvar = 0.1;            #minimum value of the variance

    # Parameters of the simulation
    NSTATES = length(w);    #number of states for the valuation system

    # Initialize valuation
    v = zeros(Int(NTIME)+1); 
    deltav = zeros(Int(NTIME)+1);
    ev = zeros(Int(NTIME)+1, NSTATES);

    # Initialize actor
    a = zeros(Int(NTIME)+1);
    deltag = zeros(Int(NTIME));
    deltah = zeros(Int(NTIME));

    # Initialize reward
    rin = 0;
    r = 0;

    
    for t = 1:NTIME
        t =  Int(t)
        # Compute reward
        # 実際のシミュレーションでは
        # rperiodはgenerate_stateの返り値を
        # rfunctionにはreff.mを入れている
        if (t >  1) & (rperiod[t]==1) 
            if (rperiod[t-1]==0)
                a[t] = a[t] + action_noise * rand(Normal(0, action_noise), 1)[1];
                a[t] = max(a[t], 0);
                r = rfun(a[t]);
            end
        end

        rin = rin + dt/tau * (rperiod[t] * r - rin);
        
        # Update valuation
        v[t+1] = v[t] + dt/tau * ( w' * sval[:,t] - v[t]);
        
        if t <= Int(MICROT/dt)
            oldv = 0;
        else
            oldv = v[Int(t-MICROT/dt)];
        end
        

        deltav[t+1] = deltav[t] + dt/taudelta * (rin + v[t] - oldv - deltav[t]);
        previous = t - (MICROT + 3 * taudelta) / dt;

        previous = Int(previous)
        if previous > 0
            ev[t+1,:] = ev[t,:] + dt/tau * (lambda * ev[previous,:] + sval[:,previous] - ev[t,:]);
        end

        if deltav[t] > 0
            term = (alphacplus * dt * deltav[t] * ev[t,:])[:,1]
            w = w + term;
        else
            term = alphacminus * dt * deltav[t] * ev[t,:];
            w = w + term;
        end

        #preventing the weights from being negative
        for c =  1:length(w)
            w[c] = max(w[c], 0);     
        end

    
        # Update actor via prediction error signals
        if t <= (NTIME - 1)
            deltag[t+1] = deltag[t] + dt/taudelta * ((rin + v[t] - a[t] * q[1] * s[t]) - varianceg * deltag[t]);
        end
        if rperiod[t] == 1
            a[t+1] = a[t];
        else
            a[t+1] = a[t] + dt/tau * (deltag[t] * q[1] * s[t] + (h * s[t] - a[t]) / varianceh);
            a[t+1] = max(a[t+1],0);
        end

        if t <= (NTIME - 1)
            deltah[t+1] = deltah[t] + dt/taudelta * (a[t] - h * s[t] - deltah[t]);
        end
        q = q[1] + alphag * dt * deltag[t] * s[t]' * a[t] * rperiod[t];
        h = h + alphah * dt * deltah[t] * s[t]' * rperiod[t];
        varianceh = varianceh + alphaprech * dt * (deltah[t]^2 - varianceh) * rperiod[t];
        varianceh = max(varianceh, minvar);
        varianceg = varianceg + alphaprecg * dt * ((deltag[t] * varianceg)^2 - varianceg) * rperiod[t];
        varianceg = max(varianceg, minvar);
       
        
    end

    return w, q, h, varianceg, varianceh, deltav, deltag, deltah, a, r, v
end



#------------------------------------------------------------------------------------------------------------------------------------------------#
# simulation part

# compute each condition
# extinction() function returns two outputs
# average, std of response intensity 30 trial bins during extinction phase
ahungry120, shungry120 = extinction(120, 0, BIN) 
asated120, ssated120 = extinction(120, 1, BIN) 
ahungry360, shungry360 = extinction(360, 0, BIN) 
asated360, ssated360 = extinction(360, 1, BIN) 

xaxis = [30, 60, 90, 120, 150, 180]

reshape((ahungry120+shungry120), length(ahungry120), 1)

MAXy = maximum(hcat(ahungry120+shungry120, 
        asated120+ssated120, 
        ahungry360+shungry360,
        asated360+ssated360)
        )

# visualize results
gr(size=(800,700))
p1 =  plot(xaxis, ahungry120', ribbon=shungry120, fillalpha=.3, xlabel="30 trials bin", ylabel="response intensity", label="Hungry", ylim=(0,MAXy), title="training 120 trials");
p1 = plot!(xaxis, asated120', ribbon=ssated120, fillalpha=.3, xlabel="30 trials bin", ylabel="response intensity", label="Sated", ylim=(0,MAXy), title="training 120 trials");

p2 =  plot(xaxis, ahungry360', ribbon=shungry360, fillalpha=.3, xlabel="30 trials bin", ylabel="response intensity", label="Hungry", ylim=(0,MAXy), title="training 120 trials");
p2 = plot!(xaxis, asated360', ribbon=ssated360, fillalpha=.3, xlabel="30 trials bin", ylabel="response intensity", label="Sated", ylim=(0,MAXy), title="training 360 trials");

plot(p1,p2)