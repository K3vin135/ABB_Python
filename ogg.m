clear
clc

% Conectar al servidor OPC
try
    disp('Conectando al servidor OPC...');
    da = opcda('localhost', 'ABB.IRC5.OPC.SERVER.DA');
    connect(da);
    disp('Conectado al servidor OPC.');
catch ME
    disp('No se pudo conectar al servidor OPC:');
    disp(ME.message);
    return; % Salir si no se puede conectar al servidor OPC
end

% Crear grupo
grp = addgroup(da, 'DemoGroup');

% Añadir ítems
ItmList = {
    '_EIA.RAPID.T_ROB1.MainModule.x', 
    '_EIA.RAPID.T_ROB1.MainModule.y', 
    '_EIA.RAPID.T_ROB1.MainModule.z',
    '_EIA.RAPID.T_ROB1.MainModule.moveRobot', 
    '_EIA.RAPID.T_ROB1.MainModule.boxCoords{1,1}',
    '_EIA.RAPID.T_ROB1.MainModule.boxCoords{1,2}',
    '_EIA.RAPID.T_ROB1.MainModule.boxCoords{1,3}',
    '_EIA.RAPID.T_ROB1.MainModule.boxCoords{2,1}',
    '_EIA.RAPID.T_ROB1.MainModule.boxCoords{2,2}',
    '_EIA.RAPID.T_ROB1.MainModule.boxCoords{2,3}',
    '_EIA.RAPID.T_ROB1.MainModule.numBoxes',
    '_EIA.RAPID.T_ROB1.MainModule.processing',
    '_EIA.RAPID.T_ROB1.MainModule.coord_x', 
    '_EIA.RAPID.T_ROB1.MainModule.coord_y', 
    '_EIA.RAPID.T_ROB1.MainModule.coord_z'
};
itm = additem(grp, ItmList);

% Inicializar Moverobot y numBoxes
Moverobot = false;
numBoxes = 0;

% Configuración del cliente TCP/IP
host = 'localhost';
port = 12345;

% Intentar abrir la conexión al servidor Python
disp('Intentando conectar al servidor Python...');
connected = false;

while ~connected
    try
        % Crear un objeto de cliente TCP/IP
        tcpipClient = tcpclient(host, port);
        data = readline(tcpipClient); % Intentar leer datos para verificar conexión
        connected = true;
        disp('Conectado al servidor Python.');
    catch
        disp('Esperando conexión al servidor Python...');
        pause(1);
    end
end

while true
    % Leer datos del servidor Python
    try
        % Leer el estado de procesamiento del robot
        processing = read(itm(end-3)).Value;
        if processing
            disp('Robot está procesando, esperando...');
            pause(1); % Esperar un segundo antes de volver a intentar
            continue;
        end
        
        if tcpipClient.BytesAvailable > 0
            data = readline(tcpipClient);
            disp(['Datos recibidos de Python: ', data]); % Mensaje de depuración
            
            % Dividir la cadena recibida en coordenadas y variables
            values = str2double(split(data, ','));
            if length(values) == 8
                % Asignar las coordenadas y variables
                x1 = values(1);
                y1 = values(2);
                z1 = values(3);
                x2 = values(4);
                y2 = values(5);
                z2 = values(6);
                Moverobot = values(7);
                numBoxes = values(8);

                % Mostrar los valores recibidos
                fprintf('Coordenadas recibidas:\n');
                fprintf('Caja 1: X=%.2f, Y=%.2f, Z=%.2f\n', x1, y1, z1);
                fprintf('Caja 2: X=%.2f, Y=%.2f, Z=%.2f\n', x2, y2, z2);
                fprintf('Moverobot: %d, numBoxes: %d\n', Moverobot, numBoxes);
            
                % Escribir Moverobot y numBoxes después de las coordenadas
                write(itm(4), Moverobot);
                disp(['Moverobot escrito: ', num2str(Moverobot)]); % Mensaje de depuración
                write(itm(11), numBoxes);
                disp(['numBoxes escrito: ', num2str(numBoxes)]); % Mensaje de depuración
            
                % Escribir nuevas coordenadas en la matriz boxCoords si moveRobot está activo
                write(itm(5), x1);
                write(itm(6), y1);
                write(itm(7), z1);
                write(itm(8), x2);
                write(itm(9), y2);
                write(itm(10), z2);

            else
                % Si los datos no tienen el formato esperado, establecer a 0
                disp("Formato de datos incorrecto");
                x1 = 0; y1 = 0; z1 = 0;
                x2 = 0; y2 = 0; z2 = 0;
            end
        else
            % Si no hay datos disponibles, establecer coordenadas a 0
            x1 = 0; y1 = 0; z1 = 0;
            x2 = 0; y2 = 0; z2 = 0;
        end
    catch ME
        disp('Error leyendo del servidor Python:');
        disp(ME.message);
        x1 = 0; y1 = 0; z1 = 0;
        x2 = 0; y2 = 0; z2 = 0;
        Moverobot = false;
    end

    % Leer y enviar las coordenadas actuales del robot al servidor Python
    try
        coord_x = read(itm(end-2)).Value;
        coord_y = read(itm(end-1)).Value;
        coord_z = read(itm(end)).Value;
        current_coords = sprintf('{"x": %f, "y": %f, "z": %f}\n', coord_x, coord_y, coord_z);
        write(tcpipClient, current_coords);
    catch ME
        disp('Error leyendo/escribiendo coordenadas actuales del robot:');
        disp(ME.message);
    end

    % Leer de nuevo para confirmar
    try
        data = read(grp);
        opcdata_boxCoords = reshape([data(5:10).Value], [3, 2])';
        % Mostrar los datos leídos
        disp('Box Coordinates:');
        disp(opcdata_boxCoords);
    catch ME
        disp('Error leyendo del servidor OPC:');
        disp(ME.message);
    end

    % Esperar un momento antes de la siguiente lectura
    pause(2);
    write(itm(4), 0);
    Moverobot=0;
    disp(['Moverobot escrito: ', num2str(Moverobot)]); % Mensaje de depuración
end

% Función para reiniciar las coordenadas de boxCoords a 0
function resetBoxCoords(itm)
    try
        for j = 1:6
            write(itm(4 + j), 0);
        end
        disp('Las coordenadas de boxCoords se han reiniciado a 0.');
    catch ME
        disp('Error reiniciando las coordenadas de boxCoords:');
        disp(ME.message);
    end
end

% Función para detener el timer y limpiar recursos
function cleanup()
    clear tcpipClient;
    disconnect(da);
    delete(da);
    disp('Conexión cerrada.');
end

% Asegurarse de que cleanup se llame al salir del script
%finishup = onCleanup(@cleanup);
