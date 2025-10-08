// store/workspaceStore.ts
import { create } from 'zustand';
import { persist } from 'zustand/middleware';

interface WorkspaceState {
  sessionId: string | null;
  history: any[]; // user_history를 반영
  artifacts: {
    product_type: string | null;
    retrieved_data: any;
    product_data: any;
    sensor_data: any;
    columns_product: any;
    cx_ward_clustering_results: any;
    cx_sna_results: any[];
    cx_lda_results: any;
    cx_opportunity_scores: any[];
    cx_cam_results: any[];
    personas: any[];
    selected_persona: any;
    service_ideas: any[];
    selected_service_idea: any;
    data_plan_for_service: any[];
    selected_data_plan_for_service: any;
    cdp_definition: any[];
    selected_cdp_definition: any;
    data_plan_recommendation_message: any;
  };
}



interface WorkspaceActions {
  setWorkspace: (workspace: Partial<WorkspaceState>) => void;
  setSessionId: (sessionId: string) => void;
  addMessageToHistory: (message: any) => void;
  resetWorkspace: () => void;
}

export const initialState: WorkspaceState = {
  sessionId: null,
  history: [],
  artifacts: {
    product_type: null,
    retrieved_data: null,
    product_data: null,
    sensor_data: null,
    columns_product: null,
    cx_ward_clustering_results: null,
    cx_sna_results: [],
    cx_lda_results: null,
    cx_opportunity_scores: [],
    cx_cam_results: [],
    personas: [],
    selected_persona: null,
    service_ideas: [],
    selected_service_idea: null,
    data_plan_for_service: [],
    selected_data_plan_for_service: null,
    cdp_definition: [],
    selected_cdp_definition: null,
    data_plan_recommendation_message: null,
  },
};

export const useWorkspaceStore = create<WorkspaceState & WorkspaceActions>()(
  persist(
    (set) => ({
      ...initialState,
      setWorkspace: (workspace) => {
        console.log('Setting workspace history:', workspace.history);
        set((state) => ({ ...state, ...workspace }));
      },
      setSessionId: (sessionId) => set({ sessionId }),
      addMessageToHistory: (message) => {
        console.log('Adding message to history:', message);
        set((state) => ({ history: [...state.history, message] }));
      },
      resetWorkspace: () => set(initialState),
    }),
    {
      name: 'workspace-storage', // 로컬 스토리지 키
      partialize: (state) => ({ sessionId: state.sessionId }), // sessionId만 영속화
    }
  )
);

if (!useWorkspaceStore.getState().sessionId) {
  useWorkspaceStore.getState().setSessionId(crypto.randomUUID());
}