package fi.bridge;

import fightinggame.Controller;

public class FIEntryPoint {
    private Controller ctrl;

    public FIEntryPoint() {
        ctrl = new Controller();
    }

    public Controller getEntryPoint() {
        return ctrl;
    }

    // Pass-throughs so Python can call ep.reset(...) and ep.step(...)
    public java.util.Map<String, Object> reset(String role) {
        return ctrl.reset(role);
    }

    public java.util.Map<String, Object> step(String action) {
        return ctrl.step(action);
    }
}


## java pipeline to run this 

cd ~/Documents/DareFightingICE-7.0
kill -9 $(lsof -t -iTCP:31415 -sTCP:LISTEN) 2>/dev/null || true
PORT=31415 ./run-macos-arm64.sh --pyftg-mode --input-sync


