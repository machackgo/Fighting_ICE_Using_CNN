package fi.bridge;

import py4j.GatewayServer;
import java.util.HashMap;
import java.util.Map;

/** Minimal Py4J gateway stub for FightingICE. Replace TODOs with real engine calls. */
public class FIEntryPoint {

    /** Very simple controller stub. Replace with real engine calls later. */
    public static class Controller {
        public Map<String, Object> reset(String role) {
            Map<String, Object> out = new HashMap<>();
            // TODO: call your real engine to start a round for role ("P1"/"P2") and read the first frame
            out.put("my_hp", 400);
            out.put("opp_hp", 400);
            out.put("my_x", 200);
            out.put("opp_x", 190);
            out.put("done", false);
            return out;
        }

        public Map<String, Object> step(String action) {
            Map<String, Object> out = new HashMap<>();
            // TODO: send action to engine, advance one frame, read state
            out.put("my_hp", 400);
            out.put("opp_hp", 399);
            out.put("my_x", 201);
            out.put("opp_x", 189);
            out.put("done", false);
            return out;
        }
    }

    private final Controller ctrl = new Controller();
    public Controller getEntryPoint() { return ctrl; }

    // Optional: expose reset/step directly on the entry point
    public Map<String, Object> reset(String role) {
        return ctrl.reset(role);
    }
    public Map<String, Object> step(String action) {
        return ctrl.step(action);
    }

    public static void main(String[] args) {
        GatewayServer server = new GatewayServer(new FIEntryPoint(), 4242);
        server.start();
        System.out.println("[FI] Py4J gateway started on port " + server.getPort());
    }
}