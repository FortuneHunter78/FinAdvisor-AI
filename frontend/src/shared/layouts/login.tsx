import type { FC, PropsWithChildren } from 'react';
import AILogo from '../ui/ai-logo';

const LoginLayout: FC<PropsWithChildren> = ({ children }) => (
  <div className='w-full h-screen flex lg:block items-center justify-center p-5 bg-linear-to-r from-gray-200 to-white dark:from-black/10 dark:to-gray-900'>
    <section className='flex w-full justify-center lg:justify-between border border-gray-400 dark:border-gray-700 h-[calc(100vh-40px)] rounded-2xl'>
      <div className='w-full lg:w-[350px] p-5 bg-white dark:bg-black rounded-2xl lg:rounded-r-none flex flex-col justify-between'>
        <main>
          <AILogo className='mb-10' />
          {children}
        </main>

        <footer className='text-xs text-gray-500 text-center mt-10 lg:mt-0'>
          <ul className='flex justify-between'>
            <li>Privacy Policy</li>
            <li>© {new Date().getFullYear()}</li>
            <li>Terms of Service</li>
          </ul>
        </footer>
      </div>

      <div className='relative w-auto hidden lg:block overflow-hidden'>
        <img
          width={800}
          height={500}
          src='/login-bg.png'
          className='w-full h-full object-cover'
        />
      </div>
    </section>
  </div>
);

export default LoginLayout;
