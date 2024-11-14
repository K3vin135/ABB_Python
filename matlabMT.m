% Import java.net.Socket and java.io.*
import java.net.Socket
import java.io.*

% Initialize socket connection
try
    socket = Socket('localhost1', 12345);
    inputStream = socket.getInputStream;
    dataInputStream = BufferedReader(InputStreamReader(inputStream));
    disp('Connected to the server.');
catch
    error('Failed to connect to the server.');
end

% Main loop
try
    while true
        % Read data from Python
        message = char(dataInputStream.readLine());
        if isempty(message)
            continue;
        end

        % Parse the coordinates
        coords = str2double(strsplit(message, ','));
        x = coords(1);
        y = coords(2);
        z = coords(3);

        % Print the coordinates
        fprintf('Received coordinates: x=%f, y=%f, z=%f\n', x, y, z);

        pause(1);  % Adjust the frequency of reading as needed
    end
catch ME
    disp('Error reading from the server.');
    disp(ME.message);
end

% Clean Up
try
    socket.close();
    disp('Disconnected from the server.');
catch
    disp('Failed to close the socket.');
end
