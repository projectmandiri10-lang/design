[Setup]
AppName=Auto Vector Sablon AI
AppVersion=1.0.0
DefaultDirName={autopf}\AutoVectorSablonAI
DefaultGroupName=Auto Vector Sablon AI
OutputDir=dist-installer
OutputBaseFilename=AutoVectorSablonAI_Setup
Compression=lzma
SolidCompression=yes
WizardStyle=modern

[Files]
Source: "dist\AutoVectorSablonAI\AutoVectorSablonAI.exe"; DestDir: "{app}"; Flags: ignoreversion
Source: "assets\tools\potrace\potrace-1.16.win64\potrace.exe"; DestDir: "{app}\assets\tools\potrace"; Flags: ignoreversion
Source: "assets\*"; DestDir: "{app}\assets"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "models\*"; DestDir: "{app}\models"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\Auto Vector Sablon AI"; Filename: "{app}\AutoVectorSablonAI.exe"
Name: "{group}\Uninstall Auto Vector Sablon AI"; Filename: "{uninstallexe}"

[Run]
Filename: "{app}\AutoVectorSablonAI.exe"; Description: "Launch Auto Vector Sablon AI"; Flags: nowait postinstall skipifsilent
